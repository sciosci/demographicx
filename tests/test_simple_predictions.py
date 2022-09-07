import unittest


class SimpleClassifierTest(unittest.TestCase):
    def test_gender(self):
        from demographicx.classifier import GenderEstimator
        gender_estimator = GenderEstimator()
        prediction = gender_estimator.predict('Jim')
        self.assertGreater(prediction['male'], prediction['female'])
        self.assertGreater(prediction['male'], prediction['unknown'])

    def test_ethnicity(self):
        from demographicx.classifier import EthnicityEstimator
        ethnicity_estimator = EthnicityEstimator()
        prediction1 = ethnicity_estimator.predict('Daniel Acuna')
        prediction2 = ethnicity_estimator.predict('Jim Yi')
        prediction3 = ethnicity_estimator.predict('Joseph Biden')
        self.assertGreater(prediction1['hispanic'], prediction1['white'])
        self.assertGreater(prediction2['asian'], prediction2['white'])
        self.assertGreater(prediction3['white'], prediction3['asian'])


if __name__ == '__main__':
    unittest.main()
