import unittest

import torch

from utils.ctc_alignment import ctc_forced_align


class CtcForcedAlignmentTest(unittest.TestCase):
    def test_empty_target_aligns_all_frames_to_blank(self):
        log_probs = torch.log_softmax(torch.randn(1, 8, 10), dim=-1)

        alignment = ctc_forced_align(
            log_probs,
            torch.empty((1, 0), dtype=torch.long),
            torch.tensor([8]),
            torch.tensor([0]),
        )

        self.assertEqual(alignment.shape, (1, 8))
        self.assertTrue(torch.equal(alignment, torch.zeros((1, 8), dtype=torch.long)))

    def test_empty_emission_returns_empty_alignment(self):
        alignment = ctc_forced_align(
            torch.empty((1, 0, 10)),
            torch.tensor([[1]], dtype=torch.long),
            torch.tensor([0]),
            torch.tensor([1]),
        )

        self.assertEqual(alignment.shape, (1, 0))


if __name__ == "__main__":
    unittest.main()
