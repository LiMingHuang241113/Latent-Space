import numpy as np
import mediapipe as mp
import cv2
import librosa
import sounddevice as sd

class PoseExtractor:
    def __init__(self, cfg):
        self.pose = mp.solutions.pose.Pose(model_complexity=0)
        self.prev = None
        self.alpha = cfg["features"]["smooth_alpha"]

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            self.prev = None
            return {"presence": False}

        lm = res.pose_landmarks.landmark

        # centroid from shoulders/hips
        idx = [11, 12, 23, 24]
        xs = [lm[i].x for i in idx]; ys = [lm[i].y for i in idx]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))

        facing = float(np.clip(1.0 - abs(lm[11].z - lm[12].z), 0, 1))
        speed = 0.0
        a = self.alpha  # smoothing factor

        # wrists
        lx, ly = lm[15].x, lm[15].y
        rx, ry = lm[16].x, lm[16].y

        if self.prev is not None:
            # smooth centroid
            cx = a * cx + (1 - a) * self.prev["cx"]
            cy = a * cy + (1 - a) * self.prev["cy"]
            dx, dy = cx - self.prev["cx"], cy - self.prev["cy"]
            speed = float(np.hypot(dx, dy))

            # smooth wrists
            lx = a * lx + (1 - a) * self.prev["lwrist"][0]
            ly = a * ly + (1 - a) * self.prev["lwrist"][1]
            rx = a * rx + (1 - a) * self.prev["rwrist"][0]
            ry = a * ry + (1 - a) * self.prev["rwrist"][1]

        # store full state for next frame
        self.prev = {
            "cx": cx, "cy": cy,
            "lwrist": (lx, ly),
            "rwrist": (rx, ry),
        }

        return {
            "presence": True,
            "cx": cx,
            "cy": cy,
            "speed": speed,
            "facing": facing,
            "lwrist": (float(lx), float(ly)),
            "rwrist": (float(rx), float(ry)),
        }

    def draw(self, frame):
        """Draw pose skeleton on the given frame (in-place)."""
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        # Get the latest landmarks
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return frame

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )
        return frame



class AudioExtractor:
    def __init__(self, cfg):
        self.sr = cfg["audio"]["sample_rate"]
        self.frame = int(self.sr * cfg["audio"]["frame_ms"] / 1000)
        self.last = {"loudness_db": -60.0, "centroid_hz": 0.0}
        sd.default.samplerate = self.sr
        sd.default.channels = 1
        sd.InputStream(callback=self._cb).start()

    def _cb(self, indata, frames, time_info, status):
        x = indata[:, 0].astype(np.float32)
        rms = np.sqrt(np.mean(x ** 2) + 1e-9)
        loud = 20 * np.log10(rms + 1e-6)
        spec_cent = librosa.feature.spectral_centroid(y=x, sr=self.sr).mean()
        self.last = {"loudness_db": float(loud), "centroid_hz": float(spec_cent)}

    def read_features(self):
        return self.last


class AffectEstimator:
    def __init__(self, cfg):
        pass

    def estimate(self, pose_data, audio_feats):
        facing = pose_data.get("facing", 0.5) if pose_data else 0.5
        motion = pose_data.get("speed", 0.0) if pose_data else 0.0
        loud = audio_feats.get("loudness_db", -50.0)
        loud_n = np.clip((loud + 60) / 40, 0, 1)
        arousal = float(np.clip(0.6 * loud_n + 0.4 * np.clip(motion * 8, 0, 1), 0, 1))
        valence = float(np.clip(0.3 + 0.7 * facing, 0, 1))
        return {"valence": valence, "arousal": arousal}


def pack_embedding(pose_data, aff, aud, d=128):
    vec = [
        pose_data.get("cx", 0.5) if pose_data else 0.5,
        pose_data.get("cy", 0.5) if pose_data else 0.5,
        pose_data.get("speed", 0.0) if pose_data else 0.0,
        pose_data.get("facing", 0.5) if pose_data else 0.5,
        aff["valence"], aff["arousal"],
        (aud["loudness_db"] + 60) / 60,
        aud["centroid_hz"] / 4000.0,
    ]
    vec = np.pad(np.array(vec, dtype=np.float32), (0, max(0, d - len(vec))), mode="constant")[:d]
    return vec

