import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_decoder.dataloader import MergedDataLoader


class TestMergedDataLoader:
    @pytest.fixture
    def create_loaders(self):
        # Create simple datasets and loaders
        data1 = torch.randn(10, 2)
        data2 = torch.randn(20, 2)
        targets1 = torch.zeros(10, dtype=torch.long)
        targets2 = torch.ones(20, dtype=torch.long)

        dataset1 = TensorDataset(data1, targets1)
        dataset2 = TensorDataset(data2, targets2)

        loader1 = DataLoader(dataset1, batch_size=5)
        loader2 = DataLoader(dataset2, batch_size=10)

        return loader1, loader2

    def test_merge(self, create_loaders):
        loader1, loader2 = create_loaders
        merged_loader = MergedDataLoader(loader1, loader2, prop1=0.5)

        # Count how many batches come from each loader
        count_loader1 = 0
        count_loader2 = 0
        num_batches = 0

        for _ in range(1000):
            batch = next(iter(merged_loader))
            num_batches += 1
            if batch[1].unique().item() == 0:
                count_loader1 += 1
            else:
                count_loader2 += 1

        # Assert that both loaders are being used
        assert count_loader1 > 0, "No batches were fetched from loader1"
        assert count_loader2 > 0, "No batches were fetched from loader2"

        # Test proportion roughly
        assert abs(count_loader1 - count_loader2) < 50, "Proportions of loaders are not as expected"

        # Check the length method
        assert len(merged_loader) == min(len(loader1), len(loader2)), "Length of merged loader is incorrect"
