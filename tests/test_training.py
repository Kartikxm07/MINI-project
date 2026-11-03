import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, PropertyMock, MagicMock

import joblib
import numpy as np
import pandas as pd
import torch


class TrainingSmokeTests(unittest.TestCase):
    def test_option_pipeline_and_saved_preprocessing(self):
        from train import run_pipeline
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as output:
            overall, _ = run_pipeline(n_samples=100, epochs=1, batch_size=32, output_dir=output)
            self.assertEqual(len(overall), 5)
            self.assertTrue(np.isfinite(overall[['RMSE', 'MAE', 'R2']]).all().all())
            saved = joblib.load(Path(output) / 'preprocessing.joblib')
            self.assertEqual(saved['target_normalization'], 'divide_by_strike')
            for name in ['ann_direct.pth', 'ann_resid.pth', 'lstm_model.pth', 'vol_model.pth', 'overall_metrics.csv', 'volatility_metrics.csv', 'loss_curves.png', 'overall_comparison.png', 'moneyness_comparison.png']:
                self.assertTrue((Path(output) / name).exists())

    def test_stock_engine_with_local_prices(self):
        from tensorflow.keras.models import Sequential
        from app import run_prediction
        n = 150
        close = 100 + np.arange(n) * .1 + np.sin(np.arange(n))
        frame = pd.DataFrame({'Open': close - .5, 'High': close + 1,
                              'Low': close - 1, 'Close': close, 'Volume': np.arange(n) + 1000},
                             index=pd.bdate_range('2024-01-01', periods=n))
        frame.columns = pd.MultiIndex.from_product([frame.columns, ['TCS.NS']])
        ticker = MagicMock()
        type(ticker).info = PropertyMock(side_effect=RuntimeError('Fundamentals unavailable'))
        original_fit = Sequential.fit
        training_sizes = []

        def short_fit(model, x, y, **kwargs):
            training_sizes.append((len(x), len(kwargs['validation_data'][0])))
            kwargs['epochs'] = 1
            return original_fit(model, x, y, **kwargs)

        with patch('app.yf.download', return_value=frame), patch('app.yf.Ticker', return_value=ticker), patch.object(Sequential, 'fit', short_fit):
            result = run_prediction('TCS.NS')
        self.assertEqual(training_sizes, [(45, 15), (45, 15)])
        self.assertEqual(len(result['test_dates']), 30)
        self.assertEqual(len(result['pred_close']), 30)
        self.assertEqual(result['test_dates'][0], frame.index[120].strftime('%Y-%m-%d'))
        self.assertTrue(np.isfinite(result['next_close']))
        self.assertEqual(result['mkcap'], 'N/A')


if __name__ == '__main__':
    unittest.main()
