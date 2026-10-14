import base64
import tempfile
import unittest
from pathlib import Path

import torch
from compute_sdk.context import Context
from compute_sdk.storage import (CHUNK, FileLock, atomic_json, digest, read_json,
                                 safe_path, write_chunk)
from compute_sdk.tensor_server import Session, load, publish
from compute_sdk.worker import dispatch


class CoreTests(unittest.TestCase):
    def test_chunk_resume_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = b"training data" * 50000
            value = digest(data)
            def upload(offset, chunk):
                return write_chunk(root, {"hash": value, "size": len(data), "offset": offset,
                                          "data": base64.b64encode(chunk).decode()})
            self.assertEqual(upload(0, data[:CHUNK])["offset"], CHUNK)
            self.assertEqual(upload(0, data[:CHUNK])["offset"], CHUNK)
            offset = CHUNK
            while offset < len(data):
                reply = upload(offset, data[offset:offset + CHUNK])
                offset = reply["offset"]
            self.assertTrue(reply["complete"])
            self.assertEqual((root / "blobs" / value).read_bytes(), data)
            bad = {"hash": digest(b"different"), "size": 3, "offset": 0, "data": base64.b64encode(b"bad").decode()}
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                write_chunk(root, bad)

    def test_path_escape_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "work"
            root.mkdir()
            (root / "escape").symlink_to(Path(tmp))
            for path in ("../outside", "/etc/passwd", "C:\\secret", "escape/file"):
                with self.assertRaises(ValueError):
                    safe_path(root, path)

    def test_owner_pause_and_execution_consent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = {"hash": digest(b""), "size": 0, "offset": 0, "data": ""}
            with self.assertRaises(PermissionError):
                dispatch(root, "upload", request)
            (root / "execution-enabled").touch()
            self.assertTrue(dispatch(root, "upload", request)["complete"])
            (root / "paused").touch()
            with self.assertRaises(PermissionError):
                dispatch(root, "upload", request)

    def test_checkpoint_restores_optimizer_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = Context(tmp)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            optimizer.zero_grad()
            model(torch.ones(2, 3)).sum().backward()
            optimizer.step()
            original = {key: value.clone() for key, value in model.state_dict().items()}
            context.checkpoint(model, optimizer, 17)
            fresh = torch.nn.Linear(3, 2)
            fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=.3)
            self.assertEqual(context.restore(fresh, fresh_optimizer), 17)
            for key, value in fresh.state_dict().items():
                torch.testing.assert_close(value, original[key])
            self.assertEqual(fresh_optimizer.param_groups[0]["lr"], .01)
            self.assertTrue(fresh_optimizer.state)
            pointer = read_json(Path(tmp) / "latest.json")
            weights = Path(tmp) / pointer["generation"] / "state.safetensors"
            weights.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "integrity"):
                context.restore(fresh, fresh_optimizer)

    def test_remote_linear_gradients_match_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root, root, "cpu")
            torch.manual_seed(0)
            x = torch.randn(2, 3, 4, requires_grad=True)
            w = torch.randn(5, 4, requires_grad=True)
            bias = torch.randn(5, requires_grad=True)
            forward = session.handle({"op": "linear_forward", "hash": publish(root, {"x": x, "w": w, "b": bias})})
            y = load(root, forward["hash"], "cpu")["y"]
            local = torch.nn.functional.linear(x, w, bias)
            torch.testing.assert_close(y, local)
            g = torch.randn_like(y)
            backward = session.handle({"op": "linear_backward", "graph": forward["graph"],
                                       "hash": publish(root, {"gradient": g})})
            grads = load(root, backward["hash"], "cpu")
            expected = torch.autograd.grad(local, (x, w, bias), g)
            for key, value in zip(("x", "w", "b"), expected):
                torch.testing.assert_close(grads[key], value)
            self.assertFalse(session.graphs)


if __name__ == "__main__":
    unittest.main()
