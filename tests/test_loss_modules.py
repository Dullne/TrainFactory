from datasets import Dataset
import torch
from sentence_transformers import CrossEncoder

from train_factory.losses.ranking import (
    LambdaLoss,
    ListMLELoss,
    ListwiseLoss,
    RankNetLoss,
)
from train_factory.losses.rl import DAPOLoss, DPOLoss, rank_based_reward
from train_factory.trainers.encoder.reranker_trainer import RerankerTrainer


class DummyCrossEncoder(CrossEncoder):
    def __init__(self, num_labels: int):
        torch.nn.Module.__init__(self)
        self._num_labels = num_labels

    @property
    def num_labels(self) -> int:
        return self._num_labels


def test_listwise_losses_run():
    scores = torch.tensor([[2.0, 1.0, 0.5]])
    labels = torch.tensor([[1.0, 0.0, 0.0]])

    listwise_loss = ListwiseLoss(temperature=1.0)(scores, labels)
    listmle_loss = ListMLELoss()(scores, labels)
    lambda_loss = LambdaLoss(metric="ndcg")(scores, labels)
    ranknet_loss = RankNetLoss()(scores, labels)

    for loss in [listwise_loss, listmle_loss, lambda_loss, ranknet_loss]:
        assert torch.is_tensor(loss)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_rl_losses_and_rewards_run():
    yes_logits = torch.tensor([2.5, 0.5, -0.5])
    no_logits = torch.tensor([0.1, 0.2, 0.3])
    labels = torch.tensor([1, 0, 0])

    rewards = rank_based_reward(torch.sigmoid(yes_logits - no_logits), labels)
    loss, advantages, reward_values, kl = DAPOLoss()(  # type: ignore[misc]
        yes_logits,
        no_logits,
        labels,
        return_stats=True,
    )

    assert rewards.shape == labels.shape
    assert advantages.shape == labels.shape
    assert reward_values.shape == labels.shape
    assert torch.isfinite(loss)
    assert torch.isfinite(kl)


def test_dpo_loss_runs():
    loss_fn = DPOLoss(beta=0.1, reference_free=True)
    loss, pos_score, neg_score = loss_fn(  # type: ignore[misc]
        pos_yes_logits=torch.tensor([2.0, 2.5]),
        pos_no_logits=torch.tensor([0.0, 0.1]),
        neg_yes_logits=torch.tensor([0.2, 0.4]),
        neg_no_logits=torch.tensor([0.3, 0.5]),
        return_stats=True,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(pos_score)
    assert torch.isfinite(neg_score)


def test_crossencoder_uses_sentence_transformers_losses_only():
    trainer = object.__new__(RerankerTrainer)
    fake_model = DummyCrossEncoder(num_labels=1)

    trainer.raw_config = {"reranker_loss_name": "BCEWithLogitsLoss", "loss_config": {}}
    bce_loss = trainer.create_loss_function(fake_model, None)
    assert bce_loss.__class__.__name__ == "BinaryCrossEntropyLoss"

    trainer.raw_config = {"reranker_loss_name": "MSELoss", "loss_config": {}}
    mse_loss = trainer.create_loss_function(fake_model, None)
    assert mse_loss.__class__.__name__ == "MSELoss"

    trainer.raw_config = {"reranker_loss_name": "MarginMSELoss", "loss_config": {}}
    margin_mse_loss = trainer.create_loss_function(fake_model, None)
    assert margin_mse_loss.__class__.__name__ == "MarginMSELoss"

    trainer.raw_config = {
        "reranker_loss_name": "MultipleNegativesRankingLoss",
        "loss_config": {"num_negatives": 3, "scale": 7.5},
    }
    mnr_dataset = Dataset.from_dict({
        "query": ["q1"],
        "positive": ["p1"],
        "negative_0": ["n1"],
    })
    mnr_loss = trainer.create_loss_function(fake_model, mnr_dataset)
    assert mnr_loss.__class__.__name__ == "MultipleNegativesRankingLoss"

    trainer.raw_config = {
        "reranker_loss_name": "CachedMultipleNegativesRankingLoss",
        "loss_config": {"num_negatives": 3, "scale": 7.5, "mini_batch_size": 8},
    }
    cached_mnr_loss = trainer.create_loss_function(fake_model, mnr_dataset)
    assert cached_mnr_loss.__class__.__name__ == "CachedMultipleNegativesRankingLoss"

    trainer.raw_config = {"reranker_loss_name": "RankNetLoss", "loss_config": {"sigma": 2.0}}
    ranknet_loss = trainer.create_loss_function(fake_model, None)
    assert ranknet_loss.__class__.__name__ == "RankNetLoss"

    trainer.raw_config = {"reranker_loss_name": "LambdaLoss", "loss_config": {}}
    lambda_loss = trainer.create_loss_function(fake_model, None)
    assert lambda_loss.__class__.__name__ == "LambdaLoss"

    trainer.raw_config = {"reranker_loss_name": "ListMLELoss", "loss_config": {}}
    listmle_loss = trainer.create_loss_function(fake_model, None)
    assert listmle_loss.__class__.__name__ == "ListMLELoss"

    trainer.raw_config = {"reranker_loss_name": "ListNetLoss", "loss_config": {}}
    listnet_loss = trainer.create_loss_function(fake_model, None)
    assert listnet_loss.__class__.__name__ == "ListNetLoss"

    trainer.raw_config = {"reranker_loss_name": "PListMLELoss", "loss_config": {}}
    plistmle_loss = trainer.create_loss_function(fake_model, None)
    assert plistmle_loss.__class__.__name__ == "PListMLELoss"

    trainer.raw_config = {"reranker_loss_name": "PairwiseHingeLoss", "loss_config": {}}
    assert trainer.create_loss_function(fake_model, None) is None


def test_reranker_flattens_universal_triplets_for_crossencoder_mnr():
    trainer = object.__new__(RerankerTrainer)
    trainer.raw_config = {
        "reranker_loss_name": "MultipleNegativesRankingLoss",
        "loss_config": {"num_negatives": 4},
    }

    datasets = {
        "train": Dataset.from_dict({
            "query": ["q1", "q2"],
            "positives": [["p1", "backup"], ["p2"]],
            "negatives": [["n1", "n2"], ["n3"]],
        })
    }

    prepared = trainer._prepare_datasets(datasets)
    train_dataset = prepared["train"]

    assert train_dataset.column_names == ["query", "positive", "negative_0"]
    assert train_dataset[0] == {"query": "q1", "positive": "p1", "negative_0": "n1"}
    assert train_dataset[1] == {"query": "q2", "positive": "p2", "negative_0": "n3"}


def test_reranker_mnr_falls_back_when_dataset_still_has_nested_columns():
    trainer = object.__new__(RerankerTrainer)
    trainer.raw_config = {
        "reranker_loss_name": "MultipleNegativesRankingLoss",
        "loss_config": {"num_negatives": 2},
    }
    fake_model = DummyCrossEncoder(num_labels=1)
    incompatible_dataset = Dataset.from_dict({
        "query": ["q1"],
        "positives": [["p1"]],
        "negatives": [["n1"]],
    })

    assert trainer.create_loss_function(fake_model, incompatible_dataset) is None
