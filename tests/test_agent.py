import unittest
from main import process_voice_order

class TestOrderAgent(unittest.TestCase):
    def test_valid_combo(self):
        result = process_voice_order(["mighty combo"])
        self.assertEqual(result["status"], "success")
        self.assertIn("Mighty Combo", result["voice_reply"])

    def test_invalid_item(self):
        result = process_voice_order(["beef burger"])
        self.assertEqual(result["status"], "error")
        self.assertIn("don't have", result["voice_reply"])

    def test_deal_mapping(self):
        result = process_voice_order(["zinger burger", "pepsi 250ml"])
        self.assertEqual(result["status"], "success")
        self.assertIn("Zinger Deal", result["voice_reply"])

if __name__ == "__main__":
    unittest.main()
