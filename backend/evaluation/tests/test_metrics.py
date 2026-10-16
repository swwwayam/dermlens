import unittest

from backend.evaluation.metrics import correct_legacy_ndcg, ideal_dcg, retrieval_metrics


class RetrievalMetricsTests(unittest.TestCase):
    def test_perfect_ranking(self):
        metrics = retrieval_metrics([1] * 10, total_relevant=20)
        self.assertEqual(metrics['precision_at_10'], 1.0)
        self.assertEqual(metrics['ndcg_at_10'], 1.0)

    def test_gallery_positives_define_ideal_ranking(self):
        metrics = retrieval_metrics([1, 0, 0], total_relevant=10)
        self.assertAlmostEqual(metrics['ndcg_at_10'], 1 / ideal_dcg(10))
        self.assertEqual(metrics['precision_at_10'], 0.1)

    def test_only_consumes_requested_ranks(self):
        def relevance():
            yield 1
            yield 0
            raise AssertionError('Read beyond requested ranks')
        self.assertEqual(retrieval_metrics(relevance(), 1, k=2)['precision_at_2'], 0.5)

    def test_invalid_counts_are_rejected_consistently(self):
        for count, k in [(1, 0), (1, -1), (1, 1.5), (1, True), (-1, 10), (1.5, 10), (True, 10)]:
            for operation in [lambda: ideal_dcg(count, k),
                              lambda: retrieval_metrics([1], count, k),
                              lambda: correct_legacy_ndcg(1, 0.1, count, k)]:
                with self.subTest(count=count, k=k), self.assertRaises(ValueError):
                    operation()

    def test_empty_gallery_and_no_matches(self):
        self.assertEqual(retrieval_metrics([], 0)['ndcg_at_10'], 0)

    def test_legacy_recovery(self):
        self.assertAlmostEqual(correct_legacy_ndcg(1, 0.1, 10), 1 / ideal_dcg(10))

    def test_invalid_relevance(self):
        with self.assertRaises(ValueError):
            retrieval_metrics([2], 10)
        with self.assertRaises(ValueError):
            retrieval_metrics([1, 1], 1)


if __name__ == '__main__':
    unittest.main()
