"""LIME (Local Interpretable Model-agnostic Explanations) for one agent decision.

Algorithm (Ribeiro et al., 2016), written out step by step:

1. Interpretable representation: the request becomes d binary features
   (tokens or parameters, see features.py). x' = [1, 1, ..., 1] is the original request.
2. Perturb: draw N masks z' in {0,1}^d. As in the reference `lime` text explainer, each
   sample removes a uniformly random number of features (1..d-1) at random positions.
   Sample 0 is the unperturbed request.
3. Query the black box: rebuild each perturbed request and get f(z) = P(class | request)
   from the decision probe.
4. Locality weights: pi(z') = sqrt(exp(-D(x', z')^2 / width^2)), D = cosine distance x 100,
   width = 25 (the `lime` text defaults). Samples close to the original count most.
5. Fit a weighted Ridge regression g(z') = w0 + w . z' to f(z) (Ridge alpha = 1).
6. Explanation = the weights w: how much keeping each feature raises (+) or lowers (-)
   P(class), locally around this request. The weighted R^2 of g tells you how well the
   linear surrogate fits; a low score means the explanation is a poor summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances

from ..data_loader import HttpRequest
from .features import RequestFeatures


@dataclass
class LimeExplanation:
    request_id: str
    ground_truth: str
    target: str
    granularity: str
    class_names: list[str]
    explained_class: str
    original_probs: dict[str, float]   # probe on the unperturbed request
    candidate_mass: float              # raw probability mass of the candidates
    intercept: float
    local_pred: float                  # surrogate's prediction for the original request
    score: float                       # weighted R^2 of the surrogate (fidelity)
    num_samples: int
    features: list[dict] = field(default_factory=list)  # label, text, part, param, span, weight
    agent_choice: str | None = None    # what the agent actually did (if from a trace)

    def top(self, k: int = 10) -> list[dict]:
        return sorted(self.features, key=lambda f: -abs(f["weight"]))[:k]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def sample_masks(d: int, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    masks = np.ones((num_samples, d), dtype=int)
    for i in range(1, num_samples):
        n_off = rng.integers(1, d) if d > 1 else 1
        masks[i, rng.choice(d, size=n_off, replace=False)] = 0
    return masks


def explain_request(
    req: HttpRequest,
    probe,
    granularity: str = "token",
    num_samples: int = 300,
    class_name: str | None = None,
    kernel_width: float = 25.0,
    seed: int = 0,
    agent_choice: str | None = None,
) -> LimeExplanation:
    feats = RequestFeatures(req, granularity)
    d = len(feats)
    if d == 0:
        raise ValueError(f"{req.request_id} has no user-controlled features to explain.")

    # 2. perturb
    masks = sample_masks(d, num_samples, np.random.default_rng(seed))
    # 3. query the black box
    probs, mass = probe.predict([feats.rebuild(m) for m in masks])

    classes = probe.class_names
    if class_name is None:  # explain what the agent did, else the probe's top class
        class_name = agent_choice if agent_choice in classes else classes[int(probs[0].argmax())]
    k = classes.index(class_name)
    y = probs[:, k]

    # 4. locality weights
    dist = cosine_distances(masks, masks[:1]).ravel() * 100
    weights = np.sqrt(np.exp(-(dist ** 2) / kernel_width ** 2))

    # 5. weighted linear surrogate
    surrogate = Ridge(alpha=1.0, fit_intercept=True)
    surrogate.fit(masks, y, sample_weight=weights)

    feature_rows = [
        {"label": f.label, "text": f.text, "part": f.part, "param": f.param_name,
         "param_idx": f.param_idx, "span": f.span, "weight": float(w)}
        for f, w in zip(feats.features, surrogate.coef_)
    ]
    return LimeExplanation(
        request_id=req.request_id,
        ground_truth=req.label,
        target=probe.target,
        granularity=granularity,
        class_names=classes,
        explained_class=class_name,
        original_probs={c: float(p) for c, p in zip(classes, probs[0])},
        candidate_mass=float(mass[0]),
        intercept=float(surrogate.intercept_),
        local_pred=float(surrogate.predict(masks[:1])[0]),
        score=float(surrogate.score(masks, y, sample_weight=weights)),
        num_samples=num_samples,
        features=feature_rows,
        agent_choice=agent_choice,
    )
