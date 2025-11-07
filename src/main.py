import asyncio, json, time
import cv2, numpy as np
import yaml
from .features import PoseExtractor, AudioExtractor, AffectEstimator, pack_embedding
from .transport import ObservationServer
from .privacy import Masker, InOptOut

# Load config
with open('config.yaml','r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f) or {}

# Init modules
pose = PoseExtractor(CFG)
audio = AudioExtractor(CFG)
affect = AffectEstimator(CFG)
masker = Masker(CFG)
optout = InOptOut(CFG)
server = ObservationServer(CFG)

# Camera
cap = cv2.VideoCapture(CFG['video']['index'])
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG['video']['width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG['video']['height'])
cap.set(cv2.CAP_PROP_FPS,          CFG['video']['fps'])

# art state 
art_canvas = None               
last_pts = {"L": None, "R": None} 

def hsv_to_bgr(h, s=255, v=255):
    return tuple(int(c) for c in cv2.cvtColor(
        np.uint8([[[int(h) % 180, int(s), int(v)]]]), cv2.COLOR_HSV2BGR
    )[0,0])

async def loop():
    await server.start()
    t_prev = time.time()
    send_interval = 1.0 / max(1e-6, CFG.get('network', {}).get('send_hz', 25))
    embedding_dim = CFG.get('features', {}).get('embedding_dim', 128)

    if not cap.isOpened():
        print("[ERROR] Camera failed to open. Check CFG['video']['index'] and permissions.")
        return

    cv2.namedWindow('Camera Feed — Observing Agent')
    print("Ready. Press SPACE to start, or ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            await asyncio.sleep(0.01)
            continue
        splash = frame.copy()
        cv2.putText(splash, "Press SPACE to start  |  ESC to quit",
                    (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3, cv2.LINE_AA)
        cv2.imshow('Camera Feed — Observing Agent', splash)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == 32:  # SPACE
            print("Starting main loop...")
            break
        await asyncio.sleep(0.01)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            await asyncio.sleep(0.01)
            continue

        if art_canvas is None:
            h, w = frame.shape[:2]
            globals()['art_canvas'] = np.zeros((h, w, 3), dtype=np.uint8)

        # Perception
        pose_data = pose.process(frame) or {'presence': False}
        presence = bool(pose_data.get('presence', False))
        audio_feats = (audio.read_features() or {})
        audio_feats.setdefault('loudness_db', -60.0)
        audio_feats.setdefault('centroid_hz', 0.0)
        audio_feats.setdefault('tempo_bpm', 0)

        # Affect
        aff = affect.estimate(pose_data, audio_feats) or {'valence': 0.5, 'arousal': 0.5}

        # Privacy: opt-out
        cx = pose_data.get('cx', 0.5)
        cy = pose_data.get('cy', 0.5)
        if presence and optout.in_region(cx, cy):
            presence = False

        # Embedding
        try:
            emb = pack_embedding(pose_data, aff, audio_feats, d=embedding_dim)
        except Exception as e:
            print("[WARN] pack_embedding failed:", e)
            emb = np.zeros((embedding_dim,), dtype=np.float32)

        if presence and art_canvas is not None:
            H, W = art_canvas.shape[:2]

            lw = pose_data.get("lwrist")  
            rw = pose_data.get("rwrist")

            # color from affect; thickness from motion
            hue = int(aff.get("valence", 0.5) * 179)
            val = int(80 + aff.get("arousal", 0.5) * 175)
            col = hsv_to_bgr(hue, 255, min(255, val))
            thick = max(2, int(1 + pose_data.get("speed", 0.0) * 20))

            def to_px(pt):
                if not pt: return None
                x = int(pt[0] * W); y = int(pt[1] * H)
                return (x, y)

            L = to_px(lw)
            R = to_px(rw)

            if L and last_pts["L"]:
                cv2.line(art_canvas, last_pts["L"], L, col, thick, cv2.LINE_AA)
            if R and last_pts["R"]:
                cv2.line(art_canvas, last_pts["R"], R, col, thick, cv2.LINE_AA)

            last_pts["L"] = L
            last_pts["R"] = R
        else:
            last_pts["L"] = last_pts["R"] = None

        # fade the canvas slightly each frame to create trails
        if art_canvas is not None:
            art_canvas[:] = cv2.addWeighted(art_canvas, 0.99, np.zeros_like(art_canvas), 0, 0)

        # Broadcast throttle
        now = time.time()
        if now - t_prev >= send_interval:
            msg = {
                't': now,
                'presence': presence,
                'pose': {
                    'cx': pose_data.get('cx', None),
                    'cy': pose_data.get('cy', None),
                    'speed': pose_data.get('speed', None),
                    'facing': pose_data.get('facing', None),
                } if presence else None,
                'wrist': {
                    'L': pose_data.get('lwrist', None),
                    'R': pose_data.get('rwrist', None),
                } if presence else None,
                'affect': {
                    'valence': float(aff.get('valence', 0.5)),
                    'arousal': float(aff.get('arousal', 0.5)),
                },
                'audio': audio_feats,
                'keywords': [],
                'embedding': emb.tolist(),
            }

            await server.broadcast(json.dumps(msg))
            t_prev = now

        # Draw skeleton on a copy of the camera
        cam_view = frame.copy()
        if CFG.get('video', {}).get('face_mask', False):
            cam_view = masker.blur_faces(cam_view)
        try:
            pose.draw(cam_view)
        except Exception:
            pass

        # Windows 
        cv2.imshow('Camera Feed — Observing Agent', cam_view)
        if art_canvas is not None:
            cv2.imshow('Movement Art — Reflecting Output', art_canvas)

        await asyncio.sleep(1.0/30)

        # Exit with ESC
        if cv2.waitKey(1) & 0xFF == 27:

            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(loop())
