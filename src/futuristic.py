# src/futuristic.py
import asyncio, json, math, random, time
import numpy as np
import cv2
import websockets

WS_URI = "ws://127.0.0.1:8765/observations"

def hsv_to_bgr(h, s=255, v=255):
    return tuple(int(c) for c in cv2.cvtColor(
        np.uint8([[[int(h)%180, int(s), int(v)]]]), cv2.COLOR_HSV2BGR
    )[0,0])

class Particle:
    __slots__ = ("x","y","vx","vy","life","maxlife","h","sat","val","size")
    def __init__(self, x, y, hue, density, speed):
        ang = random.random()*2*math.pi
        spd = (0.4 + random.random()*0.6) * speed
        self.vx = math.cos(ang)*spd
        self.vy = math.sin(ang)*spd
        self.x, self.y = float(x), float(y)
        self.maxlife = 30 + int(random.random()*60) * (0.5 + density)
        self.life = self.maxlife
        self.h = hue
        self.sat = 255
        self.val = 180
        self.size = 2 + int(3*density)

    def step(self, w, h, arousal):
        self.vx += (random.random()-0.5)*0.2*arousal
        self.vy += (random.random()-0.5)*0.2*arousal
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        if self.x<0 or self.x>=w: self.vx*=-0.8; self.x = np.clip(self.x, 0, w-1)
        if self.y<0 or self.y>=h: self.vy*=-0.8; self.y = np.clip(self.y, 0, h-1)

    def draw(self, canvas):
        col = hsv_to_bgr(self.h, self.sat, max(80, min(255, int(self.val * (self.life/self.maxlife)))))
        cv2.circle(canvas, (int(self.x), int(self.y)), self.size, col, -1, cv2.LINE_AA)

class FuturisticScene:
    def __init__(self, w=1280, h=720):
        self.w, self.h = w, h
        self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
        self.particles = []

    def spawn_burst(self, pt, valence, arousal, density):
        if pt is None: return
        hue = int(valence * 179)
        base_speed = 1.5 + 3.0*arousal
        n = 20 + int(60 * density)
        for _ in range(n):
            self.particles.append(Particle(pt[0], pt[1], hue, density, base_speed))

    def step(self, valence, arousal):
        self.canvas = cv2.addWeighted(self.canvas, 0.94, np.zeros_like(self.canvas), 0, 0)
        keep = []
        for p in self.particles:
            p.step(self.w, self.h, arousal)
            if p.life > 0:
                p.draw(self.canvas)
                keep.append(p)
        self.particles = keep

class Predictor:
    def __init__(self):
        self.prev = {"L": None, "R": None}
        self.prev_t = None

    def predict(self, L, R, t, width, height, horizon=0.20):
        out = {"L": None, "R": None}
        if self.prev_t is None:
            self.prev_t = t
        dt = max(1e-3, t - self.prev_t)

        def one(side, curr):
            prev = self.prev[side]
            if curr is None:
                if prev is None: return None
                (px, py, vx, vy) = prev
                px += vx * horizon
                py += vy * horizon
                return (int(np.clip(px * width,  0, width-1)),
                        int(np.clip(py * height, 0, height-1)))
            if prev is None:
                vx = vy = 0.0
            else:
                (px, py, vx, vy) = prev
                vx = 0.7*vx + 0.3*((curr[0]-px)/dt)
                vy = 0.7*vy + 0.3*((curr[1]-py)/dt)
            self.prev[side] = (curr[0], curr[1], vx, vy)
            px = curr[0] + vx * horizon
            py = curr[1] + vy * horizon
            return (int(np.clip(px * width,  0, width-1)),
                    int(np.clip(py * height, 0, height-1)))

        out["L"] = one("L", L)
        out["R"] = one("R", R)
        self.prev_t = t
        return out

async def client():
    print("[Futuristic] starting; connecting to", WS_URI)
    scene = FuturisticScene(1280, 720) 
    predictor = Predictor()

    waiting = True

    while True:
        try:
            async with websockets.connect(WS_URI, ping_interval=20) as ws:
                print("[Futuristic] connected.")
                waiting = False
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except asyncio.TimeoutError:
                        pass
                    else:
                        data = json.loads(msg)
                        aff = data.get("affect", {}) or {}
                        pose = data.get("pose", None)
                        wrist = data.get("wrist", None)
                        t = data.get("t", time.time())

                        def to_px(pt):
                            if not pt: return None
                            return (int(pt[0]*scene.w), int(pt[1]*scene.h))

                        L = to_px(wrist.get("L")) if wrist else None
                        R = to_px(wrist.get("R")) if wrist else None

                        if not L and pose:
                            cx = pose.get("cx", 0.5); cy = pose.get("cy", 0.5)
                            L = (int(cx*scene.w), int(cy*scene.h))

                        pred = predictor.predict(
                            (L[0]/scene.w, L[1]/scene.h) if L else None,
                            (R[0]/scene.w, R[1]/scene.h) if R else None,
                            t, scene.w, scene.h, horizon=0.20
                        )
                        valence = float(aff.get("valence", 0.5))
                        arousal = float(aff.get("arousal", 0.5))
                        density = 0.5 

                        scene.spawn_burst(pred.get("L"), valence, arousal, density)
                        scene.spawn_burst(pred.get("R"), valence, arousal, density)

                        scene.step(valence, arousal)

                    hud = scene.canvas.copy()
                    if waiting:
                        cv2.putText(hud, "Waiting for stream...", (40, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
                    cv2.imshow("Futuristic Agent — Predictive Particles", hud)
                    if cv2.waitKey(1) & 0xFF == 27:
                        cv2.destroyAllWindows()
                        return
                    await asyncio.sleep(1/60)

        except Exception as e:
            print("[Futuristic] connect error:", e)
            waiting = True
            hud = scene.canvas.copy()
            cv2.putText(hud, "Connecting...", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Futuristic Agent — Predictive Particles", hud)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                return
            await asyncio.sleep(0.5)  

if __name__ == "__main__":
    asyncio.run(client())
