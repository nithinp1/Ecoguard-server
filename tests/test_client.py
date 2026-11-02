"""Test client: streams a local WAV to /audio-stream (or an SMS to /sms-check)."""
import asyncio
import io
import json
import os
import sys
import wave

import websockets

URL = "ws://127.0.0.1:8000"


def wav_chunks(path, seconds=5):
    """Yield the WAV's audio as a stream of self-contained 5s WAV chunks."""
    with wave.open(path, "rb") as src:
        params = src.getparams()
        frames_per_chunk = src.getframerate() * seconds
        while True:
            frames = src.readframes(frames_per_chunk)
            if not frames:
                break
            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setparams(params)
                out.writeframes(frames)
            yield buf.getvalue()


async def test_audio(path):
    async with websockets.connect(f"{URL}/audio-stream") as ws:
        for chunk in wav_chunks(path):
            print(f"Sending {len(chunk)} bytes...")
            await ws.send(chunk)
            print(json.dumps(json.loads(await ws.recv()), indent=2))
        await ws.send("END")
        try:
            print("FINAL:", json.dumps(json.loads(await ws.recv()), indent=2))
        except Exception:
            pass


async def score_audio_detailed(ws, path):
    """Stream a WAV to an open /audio-stream socket; return per-chunk results and final score."""
    chunk_results = []
    max_chunk_score = 0
    for chunk in wav_chunks(path):
        await ws.send(chunk)
        raw = await ws.recv()
        try:
            res = json.loads(raw)
            chunk_results.append(res)
            sc = res.get("risk_score")
            if isinstance(sc, (int, float)) and sc > max_chunk_score:
                max_chunk_score = sc
        except Exception:
            pass
    await ws.send("END")
    try:
        final = json.loads(await ws.recv())
    except Exception:
        final = None
    return {
        "chunks": chunk_results,
        "max_chunk_score": max_chunk_score,
        "final": final,
    }


async def batch_test(root):
    wavs = sorted(
        os.path.join(dp, f)
        for dp, _, fs in os.walk(root)
        for f in fs
        if f.lower().endswith(".wav")
    )
    print(f"Found {len(wavs)} .wav file(s) in {root}\n")
    rows = []
    fps, tps, total = 0, 0, len(wavs)
    is_scam_dir = "scam" in os.path.basename(os.path.abspath(root)).lower()

    for path in wavs:
        rel = os.path.relpath(path, root)
        try:
            async with websockets.connect(f"{URL}/audio-stream") as ws:
                res = await score_audio_detailed(ws, path)
            final = res.get("final") or {}
            final_score = final.get("risk_score", 0)
            if final_score is None:
                final_score = 0
            max_score = max(final_score, res["max_chunk_score"])
            reason = (final.get("reason") or "").strip().replace("\n", " ")
            transcript = (final.get("transcript") or "").strip().replace("\n", " ")

            is_flagged = max_score >= 50
            if is_scam_dir and is_flagged:
                tps += 1
            elif not is_scam_dir and is_flagged:
                fps += 1

            status_mark = "FLAGGED" if is_flagged else "CLEAN"
            rows.append((rel, final_score, res["max_chunk_score"], status_mark, reason, transcript))
            print(f"[{status_mark:7s}] max={max_score:3d} final={final_score:3d} | {rel:20s} | {reason[:60]}", flush=True)
        except Exception as e:
            rows.append((rel, "ERR", "ERR", "ERR", str(e), ""))
            print(f"[ERR    ] {rel} : {e}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"Directory: {root}")
    print(f"Total files: {total}")
    if is_scam_dir:
        print(f"True Positives (Scams flagged >= 50): {tps} / {total} ({tps/max(1,total)*100:.1f}%)")
    else:
        print(f"False Positives (Normal flagged >= 50): {fps} / {total} ({fps/max(1,total)*100:.1f}%)")
    for rel, f_score, m_score, status, reason, tr in rows:
        print(f"{status:7s} | final={str(f_score):>3s} max_chunk={str(m_score):>3s} | {rel}")


async def test_sms(text):
    async with websockets.connect(f"{URL}/sms-check") as ws:
        await ws.send(text)
        print(json.dumps(json.loads(await ws.recv()), indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sms":
        asyncio.run(test_sms(" ".join(sys.argv[2:])))
    elif len(sys.argv) > 1 and sys.argv[1] in ("--dir", "-d") and len(sys.argv) > 2:
        asyncio.run(batch_test(sys.argv[2]))
    else:
        path = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
        asyncio.run(test_audio(path))

