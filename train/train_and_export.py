import json
import os
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

# Small dataset for demo (enough to show the enterprise workflow)
DATA = [
  # login
  ("I can't log in to my account", "login"),
  ("password reset link is not working", "login"),
  ("my account is locked please unlock", "login"),
  ("two factor code not coming through", "login"),
  ("login fails every time", "login"),
  ("it says invalid password but it's correct", "login"),

  # billing
  ("I was charged twice on my card", "billing"),
  ("refund me for this order", "billing"),
  ("my invoice is wrong", "billing"),
  ("why is there an extra fee on my bill", "billing"),
  ("I want a refund asap", "billing"),
  ("payment went through but I got billed again", "billing"),

  # shipping
  ("where is my order", "shipping"),
  ("tracking says delivered but I didn't get it", "shipping"),
  ("my package is late", "shipping"),
  ("shipping address change for my order", "shipping"),
  ("my item never arrived", "shipping"),
  ("can you update the delivery address", "shipping"),

  # technical
  ("the app keeps crashing", "technical"),
  ("I get error 500 when I try to pay", "technical"),
  ("website is not loading", "technical"),
  ("the checkout button does nothing", "technical"),
  ("your app is freezing on checkout", "technical"),
  ("I see an unexpected error", "technical"),

  # cancellation
  ("cancel my subscription", "cancellation"),
  ("I want to close my account", "cancellation"),
  ("stop my membership renewal", "cancellation"),
  ("please terminate my plan", "cancellation"),
  ("I want to cancel immediately", "cancellation"),
  ("end my subscription", "cancellation"),

  # other
  ("what are your business hours", "other"),
  ("do you have a discount code", "other"),
  ("I have a question about your product", "other"),
  ("can I speak to an agent", "other"),
  ("how do I upgrade my plan", "other"),
  ("where can I find documentation", "other"),
]

LABELS = sorted(list({label for _, label in DATA}))
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

def main():
    texts = [t for t, _ in DATA]
    y = np.array([LABEL_TO_ID[label] for _, label in DATA], dtype=np.int64)

    # Model pipeline (simple but real)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    model.fit(texts, y)

    # Export to ONNX (this is the “brain file” we deploy)
    initial_type = [("text", StringTensorType([None, 1]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model.named_steps["clf"]): {"zipmap": False}}
    )

    out_dir = "model-repo/intent_router/1"
    os.makedirs(out_dir, exist_ok=True)

    onnx_path = os.path.join(out_dir, "model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Save labels so API can map prediction index -> category name
    os.makedirs("model-repo/intent_router", exist_ok=True)
    with open("model-repo/intent_router/labels.json", "w") as f:
        json.dump({"labels": LABELS, "label_to_id": LABEL_TO_ID}, f, indent=2)

    print("✅ Exported model brain file:", onnx_path)
    print("✅ Saved labels file: model-repo/intent_router/labels.json")
    print("Labels:", LABELS)

if __name__ == "__main__":
    main()
