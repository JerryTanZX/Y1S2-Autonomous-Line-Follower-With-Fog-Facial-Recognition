import time
import socket
import struct
import cv2
import numpy as np
import multiprocessing as mp
from collections import Counter
from picamera2 import Picamera2
from gpiozero import PWMOutputDevice, DigitalOutputDevice

# ==========================================
# 🌐 FACE GATE NETWORK (Pi -> Mac)
# ==========================================
MAC_IP = "10.37.162.197"
MAC_PORT = 5002
FACE_GATE_TRIGGER_SYMBOLS = {"Fingerprint", "QR Code"}
FACE_GATE_TX_SIZE = (480, 360)          # Keep 4:3 ratio to avoid face distortion
FACE_GATE_JPEG_QUALITY = 70
FACE_GATE_SEND_INTERVAL = 0.02
FACE_GATE_CONNECT_TIMEOUT_SEC = 5.0
FACE_GATE_RECV_TIMEOUT_SEC = 2.0


class LineReader:
    def __init__(self, sock):
        self.sock = sock
        self.buf = bytearray()

    def recv_line(self, timeout_sec=None):
        deadline = None if timeout_sec is None else (time.time() + timeout_sec)
        while True:
            idx = self.buf.find(b"\n")
            if idx != -1:
                line = self.buf[:idx]
                del self.buf[: idx + 1]
                return line.decode("utf-8", errors="ignore").strip()

            if deadline is None:
                self.sock.settimeout(None)
            else:
                remain = deadline - time.time()
                if remain <= 0:
                    return None
                self.sock.settimeout(remain)

            try:
                chunk = self.sock.recv(1024)
            except socket.timeout:
                return None

            if not chunk:
                return ""
            self.buf.extend(chunk)


def run_face_gate_session(picam2):
    """
    Stop-and-wait face gate:
    - Pi streams JPEG frames to Mac
    - Mac responds with 'wait' / 'resume' / 'quit'
    - Pi returns True only when 'resume' is received
    """
    sock = None
    try:
        print(f"   -> 🧠 FACE GATE: Connecting to Mac {MAC_IP}:{MAC_PORT} ...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(FACE_GATE_CONNECT_TIMEOUT_SEC)
        sock.connect((MAC_IP, MAC_PORT))
        sock.settimeout(None)
        reader = LineReader(sock)
        print("   -> 📡 FACE GATE: Connected. Streaming frames to Mac...")

        while True:
            frame = picam2.capture_array()
            if frame is None:
                time.sleep(0.005)
                continue
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if FACE_GATE_TX_SIZE is not None:
                frame_bgr = cv2.resize(frame_bgr, FACE_GATE_TX_SIZE, interpolation=cv2.INTER_LINEAR)

            ok, encoded = cv2.imencode(
                ".jpg",
                frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), FACE_GATE_JPEG_QUALITY],
            )
            if not ok:
                time.sleep(0.005)
                continue

            jpeg_bytes = encoded.tobytes()
            payload = struct.pack(">I", len(jpeg_bytes)) + jpeg_bytes
            sock.sendall(payload)

            cmd = reader.recv_line(timeout_sec=FACE_GATE_RECV_TIMEOUT_SEC)
            if cmd == "":
                print("   -> ⚠️ FACE GATE: Mac disconnected during session.")
                return False
            if cmd is not None:
                cmd = cmd.lower().strip()
                if cmd == "resume":
                    print("   -> ✅ FACE GATE: Resume command received from Mac.")
                    return True
                if cmd == "quit":
                    print("   -> ⛔ FACE GATE: Quit command received from Mac.")
                    return False

            if FACE_GATE_SEND_INTERVAL > 0:
                time.sleep(FACE_GATE_SEND_INTERVAL)

    except Exception as exc:
        print(f"   -> ⚠️ FACE GATE ERROR: {exc}")
        return False
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

# ==========================================
# 🧠 PROCESS 2: THE THINKER (Runs on its own CPU Core)
# ==========================================
def thinker_process(frame_queue, result_queue):
    print("[THINKER] Booting up on a dedicated CPU core...")
    
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    image_files = {
        "Fingerprint": ("/home/jerry5778/Pictures/FingerPrint2.png", 30),
        "Warning": ("/home/jerry5778/Pictures/Caution2.png", 30),
        "Push Button": ("/home/jerry5778/Pictures/Button2.png", 60)
    }

    templates = {}
    for name, (filename, threshold) in image_files.items():
        try:
            img = cv2.imread(filename, 0)
            if img is not None:
                kp, des = orb.detectAndCompute(img, None)
                templates[name] = {"image": img, "keypoints": kp, "descriptors": des, "min_matches": threshold}
        except Exception:
            pass

    try:
        template_img = cv2.imread('/home/jerry5778/Pictures/Recycle.png', 0)
        if template_img is not None:
            _, temp_thresh = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY_INV)
            temp_contours, _ = cv2.findContours(temp_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            temp_contours = sorted(temp_contours, key=cv2.contourArea, reverse=True)
            template_arrow = temp_contours[0] 
    except Exception:
        pass

    print("[THINKER] Ready and waiting for frames!")

    while True:
        frame = frame_queue.get()
        if frame is None:
            break
            
        detected_symbols = []
        special_logo_rects = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray_mask = cv2.threshold(blurred_gray, 130, 255, cv2.THRESH_BINARY_INV)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)
        blurred_s = cv2.GaussianBlur(s, (5, 5), 0)
        _, sat_mask = cv2.threshold(blurred_s, 100, 255, cv2.THRESH_BINARY)

        # --- BRAIN 1: ORB MATCHING ---
        kp_frame, des_frame = orb.detectAndCompute(gray, None)
        if des_frame is not None:
            for name, data in templates.items():
                matches = bf.match(data["descriptors"], des_frame)
                if len(matches) > data["min_matches"]:
                    src_pts = np.float32([data["keypoints"][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        h_img, w_img = data["image"].shape
                        pts = np.float32([[0, 0], [0, h_img - 1], [w_img - 1, h_img - 1], [w_img - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        
                        if cv2.isContourConvex(np.int32(dst)):
                            bx, by, bw, bh = cv2.boundingRect(np.int32(dst))
                            if (bw * bh) > 1500:  
                                special_logo_rects.append((bx, by, bw, bh))
                                detected_symbols.append(name)

        # --- BRAIN 2: QR MARKER ---
        contours_qr, hierarchy = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        found_markers = []
        if hierarchy is not None:
            for i, contour in enumerate(contours_qr):
                if cv2.contourArea(contour) > 500: 
                    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                    if len(approx) == 4 and hierarchy[0][i][2] != -1:
                        child_approx = cv2.approxPolyDP(contours_qr[hierarchy[0][i][2]], 0.04 * cv2.arcLength(contours_qr[hierarchy[0][i][2]], True), True)
                        if len(child_approx) == 4: found_markers.append(approx)

        if len(found_markers) >= 3:
            x, y, w, h = cv2.boundingRect(np.vstack(found_markers))
            special_logo_rects.append((x, y, w, h))
            detected_symbols.append("QR Code")

        # --- BRAIN 3: RECYCLE LOGO ---
        contours_recy, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_arrows = []
        try:
            for contour in contours_recy:
                if cv2.contourArea(contour) < 500: continue 
                match_score = cv2.matchShapes(template_arrow, contour, cv2.CONTOURS_MATCH_I1, 0.0)

                if match_score < 0.5:
                    found_arrows.append(contour)
                    
            if len(found_arrows) >= 3:
                x, y, w, h = cv2.boundingRect(np.vstack(found_arrows))
                special_logo_rects.append((x, y, w, h))
                detected_symbols.append("Recycle Logo")
        except NameError:
            pass

        # --- BRAIN 4: COLORED SHAPES (Detects Arrows too!) ---
        contours_shapes, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_shapes:
            area = cv2.contourArea(contour)
            if area < 800: continue 

            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = int(x + (w/2)), int(y + (h/2))

            if any(sx < cx < (sx + sw) and sy < cy < (sy + sh) for (sx, sy, sw, sh) in special_logo_rects):
                continue 

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            vertices = len(approx)
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            aspect_ratio = float(w) / h if h > 0 else 0
            shape_name = "Unknown"

            # --- EXTENDED SHAPE DETECTION LOGIC ---
            if vertices == 4 and 19000 <= area <= 27000:
                if 0.45 <= extent <= 0.55: shape_name = "Diamond"
                elif extent > 0.85: shape_name = "Square" if 0.90 <= aspect_ratio <= 1.10 else "Rectangle"
                else: shape_name = "Trapezoid"
            elif vertices == 8 and 18000 <= area <= 22000: shape_name = "Octagon"
            elif vertices == 12 and 0.85 < solidity < 0.95 and 18000 <= area <= 22000: shape_name = "Cross"
            elif vertices in [7, 8, 9, 10] and 0.55 < solidity < 0.65: 
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    moment_cx, moment_cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                    extLeft, extRight = tuple(contour[contour[:, :, 0].argmin()][0]), tuple(contour[contour[:, :, 0].argmax()][0])
                    extTop, extBot = tuple(contour[contour[:, :, 1].argmin()][0]), tuple(contour[contour[:, :, 1].argmax()][0])

                    if w > h:
                        shape_name = "Arrow: LEFT" if abs(extRight[0] - moment_cx) > abs(moment_cx - extLeft[0]) else "Arrow: RIGHT"
                    else:
                        shape_name = "Arrow: DOWN" if abs(moment_cy - extTop[1]) > abs(extBot[1] - moment_cy) else "Arrow: UP"
            elif vertices == 10 and solidity < 0.55 and 8000 <= area <= 10000: shape_name = "Star"
            elif vertices > 6 and 14000 <= area <= 19000:
                # Note: A full circle will also trigger solidity > 0.90 here!
                if solidity > 0.90: shape_name = "Half Circle"
                elif 0.70 <= solidity <= 0.85: shape_name = "Partial Circle"

            if shape_name != "Unknown":
                detected_symbols.append(shape_name)

        if len(detected_symbols) > 0:
            unique_symbols = list(set(detected_symbols))
            result_queue.put(unique_symbols)


# ==========================================
# 🚀 PROCESS 1: THE DRIVER (Main Process)
# ==========================================
if __name__ == '__main__':
    frame_queue = mp.Queue(maxsize=1) 
    result_queue = mp.Queue()
        
    thinker = mp.Process(target=thinker_process, args=(frame_queue, result_queue))
    thinker.daemon = True 
    thinker.start()

    LEFT_BASE  = 0.22
    RIGHT_BASE = 0.22
    Kp = 0.009
    Kd = 0.005
    Ki = 0.0
    
    BLACK_THRESHOLD = 80
    STOP_LINE_AREA = 13500

    motor_left_speed = PWMOutputDevice(25)
    motor_left_in1 = DigitalOutputDevice(23)
    motor_left_in2 = DigitalOutputDevice(24)
    motor_right_speed = PWMOutputDevice(22)
    motor_right_in3 = DigitalOutputDevice(17)
    motor_right_in4 = DigitalOutputDevice(27)

    def set_motor(left_val, right_val):
        left_val = max(min(left_val, 1.0), -1.0)
        right_val = max(min(right_val, 1.0), -1.0)
        if left_val >= 0:
            motor_left_in1.on(); motor_left_in2.off()
        else:
            motor_left_in1.off(); motor_left_in2.on()
        motor_left_speed.value = abs(left_val)
        
        if right_val >= 0:
            motor_right_in3.on(); motor_right_in4.off()
        else:
            motor_right_in3.off(); motor_right_in4.on()
        motor_right_speed.value = abs(right_val)

    def stop():
        set_motor(0, 0)

    print("Starting Raspberry Pi Camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "XRGB8888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    
    print("=== PID 巡线 + MULTIPROCESSING VISION 启动 ===")
    last_error = 0
    I = 0
    
    ignore_vision_until = 0.0
    
    # 🏹 ARROW MEMORY VARIABLES
    current_direction = "UP" # Default state
    arrow_expiry_time = 0.0

    # 🌈 NEW: SHORTCUT STATE VARIABLES
    is_on_shortcut = False
    shortcut_exit_direction = "UP"
    
    # ⬛ NEW: BLACK SHORTCUT STATE VARIABLES
    is_on_black_shortcut = False
    black_shortcut_exit_direction = "UP"
    intersection_cooldown = 0.0  # Prevents double-counting the same intersection!

    try:
        while True:
            raw_frame = picam2.capture_array()
            
            if time.time() > ignore_vision_until:
                if not frame_queue.full():
                    thinker_roi = raw_frame[0:350, 0:640]
                    try:
                        frame_queue.put_nowait(thinker_roi)
                    except mp.queues.Full:
                        pass 
            else:
                while not result_queue.empty():
                    result_queue.get()

            # --- ARROW MEMORY EXPIRY CHECK ---
            if time.time() > arrow_expiry_time:
                current_direction = "UP" # Forget the arrow and return to normal

            if time.time() > ignore_vision_until:
                while not result_queue.empty():
                    found_symbols = result_queue.get()
                    action_taken = False
                    
                    for sym in found_symbols:
                        print(f"\n🧠 [THINKER SAYS]: I see a {sym}!!")
                        
                        if sym in ["Push Button", "Warning"]:
                            print("   -> 🛑 ACTION: Stopping for 3 seconds!")
                            stop()
                            time.sleep(3.0)
                            action_taken = True
                            break
                            
                        elif sym == "Recycle Logo":
                            print("   -> ♻️ ACTION: Spinning 360 degrees!")
                            set_motor(0.8, -0.8) 
                            time.sleep(1.8)      
                            stop()
                            action_taken = True
                            break
                            
                        # 🏹 FEATURE 2: Arrow Memory Logging
                        elif "Arrow" in sym:
                            direction = sym.split(": ")[1]
                            print(f"   -> 🔀 ACTION: Remembering to turn {direction} at the next fork!")
                            current_direction = direction
                            arrow_expiry_time = time.time() + 2.0 # Remember for 2 seconds
                            break
                            
                        elif sym in FACE_GATE_TRIGGER_SYMBOLS:
                            print("   -> 🛑 ACTION: Face gate triggered. Stop and wait for Mac resume (press 'f' on Mac).")
                            stop()
                            gate_resumed = run_face_gate_session(picam2)
                            if gate_resumed:
                                print("   -> 🚗 ACTION: Resume line-following.")
                            else:
                                print("   -> ⚠️ ACTION: No resume command received. Continuing with normal loop.")
                            action_taken = True
                            break
                            
                    if action_taken:
                        print(f"   -> ⏱️ IGNORING VISION FOR 2 SECONDS...")
                        ignore_vision_until = time.time() + 2.0
                        while not result_queue.empty(): result_queue.get()
                        last_error = 0
                        I = 0
                        break 
            
            # ==================================================
            # --- D. THE PATHFINDER LINE FOLLOWER ---
            # ==================================================
            driver_frame = cv2.resize(raw_frame, (320, 240))
            roi_line = driver_frame[115:175, 40:280]
            
            # 1. 🌈 COLORED LINE MASKING (HSV)
            hsv_roi = cv2.cvtColor(roi_line, cv2.COLOR_BGR2HSV)
            
            # --- STANDARD YELLOW MASK ---
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
            
            # --- STANDARD RED MASK (Requires two masks to handle the wrap-around) ---
            # Lower Red Range
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
            
            # Upper Red Range
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([179, 255, 255])
            mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
            
            # Combine the two Red masks
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Combine Yellow and Red to find the colored tape
            color_mask = cv2.bitwise_or(mask_yellow, mask_red)
            color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_color_contours = [c for c in color_contours if cv2.contourArea(c) > 2000]
            
            # 2. 🖤 BLACK LINE MASKING
            gray = cv2.cvtColor(roi_line, cv2.COLOR_BGR2GRAY)
            _, black_mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
            black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_black_contours = [c for c in black_contours if cv2.contourArea(c) > 150]

            # --- DECISION ENGINE ---
            target_contour = None
            
            # Priority 1: If we see a colored shortcut, ignore black entirely!
            if len(valid_color_contours) > 0:
                target_contour = max(valid_color_contours, key=cv2.contourArea)
                cv2.imshow("Line Follower ROI", color_mask)
                
                # 🧠 STATE MACHINE: We just entered the color line!
                if not is_on_shortcut:
                    is_on_shortcut = True
                    
                    # Calculate exactly where the color line is on the screen!
                    M_color = cv2.moments(target_contour)
                    if M_color["m00"] != 0:
                        entry_cx = int(M_color["m10"] / M_color["m00"])
                        
                        # The screen is 320px wide. Center is 160.
                        if entry_cx > 100:
                            print("\n🌈 [STATE] Color tape found on the RIGHT! Entering shortcut.")
                            shortcut_exit_direction = "RIGHT" 
                        else:
                            print("\n🌈 [STATE] Color tape found on the LEFT! Entering shortcut.")
                            shortcut_exit_direction = "LEFT"
                
            # Priority 2: Follow the black line
            elif len(valid_black_contours) > 0:
                target_contour = max(valid_black_contours, key=cv2.contourArea)
                cv2.imshow("Line Follower ROI", black_mask)
                
                # 🧠 STATE MACHINE: We just dropped off the color line!
                if is_on_shortcut:
                    is_on_shortcut = False
                    print(f"\n🛣️ [STATE] Exiting Shortcut! Injecting {shortcut_exit_direction} memory!")
                    
                    # Force the robot to turn in the exit direction!
                    current_direction = shortcut_exit_direction
                    current_time = time.time()
                    arrow_expiry_time = current_time + 3.0 # Give it 3 seconds to execute the merge
                    
                    # 🛡️ THE FIX: Blind the Black Shortcut brain for 3 seconds!
                    # This lets the robot steer through the wide merge blob without accidentally 
                    # triggering a new "Black Shortcut Entry" state.
                    intersection_cooldown = current_time + 3.0

            # --- STEERING MATH ---
            if target_contour is not None:
                area = cv2.contourArea(target_contour)
                
                if area > STOP_LINE_AREA:
                    print(f"🛑 检测到终点线！面积: {area} > {STOP_LINE_AREA}")
                    stop()
                    break 
                
                M = cv2.moments(target_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    
                    x, y, w, h = cv2.boundingRect(target_contour)
                    
                    # --- CALCULATE TRUE HORIZONTAL WIDTH ---
                    iso_mask = np.zeros((60, 240), dtype=np.uint8) # Size of the roi_line
                    cv2.drawContours(iso_mask, [target_contour], -1, 255, thickness=cv2.FILLED)
                    true_width = np.max(np.sum(iso_mask == 255, axis=1))
                    
                    # TRIGGER USING TRUE_WIDTH INSTEAD OF W!
                    if true_width > 100:  
                        current_time = time.time()
                        
                        # --- 1. ARE WE EXITING A BLACK SHORTCUT? ---
                        if is_on_black_shortcut and current_time > intersection_cooldown:
                            print(f"\n⬛ [STATE] Exiting Black Shortcut! Injecting {black_shortcut_exit_direction} memory!")
                            current_direction = black_shortcut_exit_direction
                            arrow_expiry_time = current_time + 2.0 
                            is_on_black_shortcut = False
                            intersection_cooldown = current_time + 3.0 
                        
                        # --- 2. ARE WE ENTERING A BLACK SHORTCUT? ---
                        elif not is_on_black_shortcut and current_direction in ["LEFT", "RIGHT"] and current_time > intersection_cooldown:
                            print(f"\n⬛ [STATE] Entering Black Shortcut! Will automatically exit {current_direction} later.")
                            is_on_black_shortcut = True
                            black_shortcut_exit_direction = current_direction
                            intersection_cooldown = current_time + 3.0
                        
                        # --- 3. EXECUTE THE TURN ---
                        # Note: We still use standard 'w' and 'x' here because they perfectly represent 
                        # the extreme left and right physical edges of the track for the steering target!
                        if current_direction == "LEFT":
                            cx = x           
                        elif current_direction == "RIGHT":
                            cx = (x + w)     
                        elif current_direction == "UP":
                            extTop = tuple(target_contour[target_contour[:, :, 1].argmin()][0])
                            cx = extTop[0]
                    
                    # Calculate the error based on our chosen cx target
                    error = cx - 120
                    
                    P = error
                    I += error
                    D = error - last_error
                    
                    correction = (Kp * P) + (Ki * I) + (Kd * D)
                    last_error = error
                    
                    left_motor_speed = LEFT_BASE + correction
                    right_motor_speed = RIGHT_BASE - correction
                    set_motor(left_motor_speed, right_motor_speed)
                    
                    # Blue dot if color line, Red dot if black line
                    dot_color = (255, 0, 0) if len(valid_color_contours) > 0 else (0, 0, 255)
                    cv2.circle(driver_frame, (cx + 40, int(M["m01"] / M["m00"]) + 115), 5, dot_color, -1)

            cv2.imshow("Driver Full View (Shrunk to 320p)", driver_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nUser Interrupted!")
    finally:
        stop()
        picam2.stop()
        frame_queue.put(None) 
        thinker.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("Shutdown Complete.")
