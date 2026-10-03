import json
import math
import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict


LABELS = ("with_mask", "without_mask")
MIN_FEEDBACK_FOR_POLICY = 5
MAX_PENDING_PREDICTIONS = 1000
PENDING_PREDICTION_TTL_SECONDS = 900
CONFIDENCE_BUCKET_COUNT = 5


class FeedbackTokenError(ValueError):
    pass


class FeedbackLearner:
    """A small, persistent contextual bandit trained from explicit user rewards."""

    def __init__(self, policy_path):
        self.policy_path = policy_path
        self._lock = threading.Lock()
        self._q_values = {}
        self._action_counts = {}
        self._state_counts = {}
        self._total_feedback = 0
        self._pending_predictions = OrderedDict()
        self._load_policy()

    @staticmethod
    def _state_key(model_label, confidence):
        if model_label not in LABELS:
            raise ValueError("The model label must be with_mask or without_mask.")
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
            raise ValueError("Confidence must be a finite number.")
        if not 0.5 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.5 and 1.0.")
        bucket = min(
            CONFIDENCE_BUCKET_COUNT - 1,
            int((confidence - 0.5) * CONFIDENCE_BUCKET_COUNT * 2 + 1e-9),
        )
        return f"{model_label}:{bucket}"

    @staticmethod
    def _empty_action_values():
        return {label: 0.0 for label in LABELS}

    def _load_policy(self):
        try:
            with open(self.policy_path, encoding="utf-8") as policy_file:
                data = json.load(policy_file)
        except FileNotFoundError:
            return

        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError(f"Unsupported reinforcement-learning policy format: {self.policy_path}")

        q_values = data.get("q_values")
        action_counts = data.get("action_counts")
        state_counts = data.get("state_counts")
        total_feedback = data.get("total_feedback")
        if (
            not isinstance(q_values, dict)
            or not isinstance(action_counts, dict)
            or not isinstance(state_counts, dict)
            or not isinstance(total_feedback, int)
            or total_feedback < 0
        ):
            raise ValueError(f"Invalid reinforcement-learning policy data: {self.policy_path}")

        for state, values in q_values.items():
            if not isinstance(state, str) or not isinstance(values, dict):
                raise ValueError(f"Invalid reinforcement-learning state: {self.policy_path}")
            if set(values) != set(LABELS) or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in values.values()
            ):
                raise ValueError(f"Invalid reinforcement-learning action values: {self.policy_path}")
            self._q_values[state] = {label: float(values[label]) for label in LABELS}

        for state, counts in action_counts.items():
            if (
                state not in self._q_values
                or not isinstance(counts, dict)
                or set(counts) != set(LABELS)
                or any(not isinstance(count, int) or count < 0 for count in counts.values())
            ):
                raise ValueError(f"Invalid reinforcement-learning action counts: {self.policy_path}")
            self._action_counts[state] = dict(counts)

        for state, count in state_counts.items():
            if (
                state not in self._q_values
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError(f"Invalid reinforcement-learning feedback counts: {self.policy_path}")
            self._state_counts[state] = count

        if sum(self._state_counts.values()) != total_feedback:
            raise ValueError(f"Inconsistent reinforcement-learning feedback counts: {self.policy_path}")
        self._total_feedback = total_feedback

    def _save_policy(self):
        policy_dir = os.path.dirname(self.policy_path)
        os.makedirs(policy_dir, exist_ok=True)
        data = {
            "version": 1,
            "q_values": self._q_values,
            "action_counts": self._action_counts,
            "state_counts": self._state_counts,
            "total_feedback": self._total_feedback,
        }
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=policy_dir,
                prefix=".mask-feedback-",
                suffix=".tmp",
                delete=False,
            ) as policy_file:
                temp_path = policy_file.name
                json.dump(data, policy_file, sort_keys=True)
                policy_file.flush()
                os.fsync(policy_file.fileno())
            os.replace(temp_path, self.policy_path)
        except OSError:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def register_prediction(self, model_label, confidence):
        state = self._state_key(model_label, confidence)
        with self._lock:
            values = self._q_values.get(state, self._empty_action_values())
            feedback_count = self._state_counts.get(state, 0)
            chosen_label = model_label
            if feedback_count >= MIN_FEEDBACK_FOR_POLICY:
                best_label = max(LABELS, key=values.__getitem__)
                if values[best_label] > values[model_label]:
                    chosen_label = best_label

            now = time.monotonic()
            while self._pending_predictions:
                oldest_token, oldest_prediction = next(iter(self._pending_predictions.items()))
                if (
                    len(self._pending_predictions) <= MAX_PENDING_PREDICTIONS
                    and now - oldest_prediction["created_at"] <= PENDING_PREDICTION_TTL_SECONDS
                ):
                    break
                self._pending_predictions.pop(oldest_token)

            feedback_id = uuid.uuid4().hex
            self._pending_predictions[feedback_id] = {
                "created_at": now,
                "state": state,
                "model_label": model_label,
                "chosen_label": chosen_label,
            }
            return {
                "feedback_id": feedback_id,
                "label": chosen_label,
                "model_label": model_label,
                "adjusted": chosen_label != model_label,
                "state_feedback_count": feedback_count,
            }

    def record_feedback(self, feedback_id, actual_label):
        if actual_label not in LABELS:
            raise ValueError("Choose with_mask or without_mask as the corrected label.")
        if not isinstance(feedback_id, str):
            raise FeedbackTokenError("This prediction has expired. Wait for the next detection and try again.")

        with self._lock:
            prediction = self._pending_predictions.pop(feedback_id, None)
            if prediction is None:
                raise FeedbackTokenError("This prediction has expired or was already reviewed. Try the latest detection.")
            if time.monotonic() - prediction["created_at"] > PENDING_PREDICTION_TTL_SECONDS:
                raise FeedbackTokenError("This prediction has expired. Wait for the next detection and try again.")

            reward = 1.0 if prediction["chosen_label"] == actual_label else -1.0
            state = prediction["state"]
            action = prediction["chosen_label"]
            values = self._q_values.setdefault(state, self._empty_action_values())
            counts = self._action_counts.setdefault(state, {label: 0 for label in LABELS})
            previous_value = values[action]
            previous_action_count = counts[action]
            previous_state_count = self._state_counts.get(state, 0)
            previous_total_feedback = self._total_feedback

            counts[action] += 1
            values[action] += (reward - values[action]) / counts[action]
            self._state_counts[state] = previous_state_count + 1
            self._total_feedback += 1
            try:
                self._save_policy()
            except OSError:
                values[action] = previous_value
                counts[action] = previous_action_count
                self._state_counts[state] = previous_state_count
                self._total_feedback = previous_total_feedback
                self._pending_predictions[feedback_id] = prediction
                raise

            return {
                "reward": int(reward),
                "state_feedback_count": self._state_counts[state],
                "total_feedback": self._total_feedback,
                "policy_active": self._state_counts[state] >= MIN_FEEDBACK_FOR_POLICY,
            }
