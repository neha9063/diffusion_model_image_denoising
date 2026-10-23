"""
Comprehensive Integration & Verification Test Suite
Tests all routes, ML pipelines, forward diffusion, reverse diffusion,
uploading, and metrics calculation.
"""

import os
import unittest
import json
import io
from PIL import Image
import torch

from app import app, load_trained_model, MODEL_PATH
from diffusion import DiffusionUNet, GaussianDiffusion


class TestDiffusionDenoisingApp(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_01_model_checkpoint_exists(self):
        """Verifies trained model checkpoint exists and loads."""
        self.assertTrue(os.path.exists(MODEL_PATH), f"Model checkpoint {MODEL_PATH} must exist.")
        loaded = load_trained_model()
        self.assertTrue(loaded, "Model should load successfully into memory.")

    def test_02_page_routes(self):
        """Tests that all HTML pages render successfully (HTTP 200)."""
        routes = ["/", "/learn", "/results"]
        for r in routes:
            resp = self.client.get(r)
            self.assertEqual(resp.status_code, 200, f"Route {r} failed with status {resp.status_code}")
            self.assertIn(b"Diffusion", resp.data)

    def test_03_api_status_and_samples(self):
        """Tests /api/status and /api/samples endpoints."""
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        status_data = json.loads(resp.data)
        self.assertTrue(status_data["loaded"])
        self.assertEqual(status_data["timesteps"], 300)

        resp2 = self.client.get("/api/samples")
        self.assertEqual(resp2.status_code, 200)
        samples = json.loads(resp2.data)
        self.assertGreaterEqual(len(samples), 10, "Should have 10 sample digits available.")

    def test_04_sample_upload(self):
        """Tests selecting a pre-loaded sample digit via /api/upload."""
        resp = self.client.post("/api/upload", data={"sample_file": "digit_3.png"})
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("clean_filename", data)
        self.assertIn("data:image/png;base64,", data["preview_url"])

    def test_05_custom_file_upload(self):
        """Tests uploading a custom PNG image file."""
        # Create a dummy 50x50 test image
        img = Image.new("RGB", (50, 50), color=(200, 100, 50))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        resp = self.client.post(
            "/api/upload",
            data={"image": (img_bytes, "test_custom.png")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("clean_", data["clean_filename"])

    def test_06_forward_and_reverse_diffusion_pipeline(self):
        """
        Tests complete end-to-end forward noise injection followed by
        PyTorch U-Net reverse diffusion denoising and quantitative metrics.
        """
        # Step A: Select sample
        resp_up = self.client.post("/api/upload", data={"sample_file": "digit_7.png"})
        self.assertEqual(resp_up.status_code, 200)
        clean_file = json.loads(resp_up.data)["clean_filename"]

        # Step B: Forward diffusion at timestep t=60
        resp_fwd = self.client.post(
            "/api/forward-diffuse",
            data=json.dumps({"clean_filename": clean_file, "timestep": 60}),
            content_type="application/json"
        )
        self.assertEqual(resp_fwd.status_code, 200)
        fwd_data = json.loads(resp_fwd.data)
        self.assertTrue(fwd_data["success"])
        self.assertIn("noisy_filename", fwd_data)
        noisy_file = fwd_data["noisy_filename"]
        self.assertIn("math_info", fwd_data)

        # Step C: Reverse diffusion from t=60 down to 0
        resp_rev = self.client.post(
            "/api/reverse-diffuse",
            data=json.dumps({
                "clean_filename": clean_file,
                "noisy_filename": noisy_file,
                "timestep": 60
            }),
            content_type="application/json"
        )
        self.assertEqual(resp_rev.status_code, 200)
        rev_data = json.loads(resp_rev.data)
        self.assertTrue(rev_data["success"])
        self.assertIn("denoised_filename", rev_data)
        self.assertIn("metrics", rev_data)
        metrics = rev_data["metrics"]
        self.assertIn("psnr_db", metrics)
        self.assertIn("mse", metrics)
        self.assertIn("inference_time_ms", metrics)
        self.assertGreater(metrics["psnr_db"], 0)
        self.assertGreater(len(rev_data["snapshots"]), 0)
        print(f"\n[+] Pipeline Test Passed! PSNR: {metrics['psnr_db']} dB, MSE: {metrics['mse']}, Latency: {metrics['inference_time_ms']} ms")


if __name__ == "__main__":
    unittest.main()
