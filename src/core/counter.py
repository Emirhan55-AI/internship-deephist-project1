from typing import Any


class StudentCounter:
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        max_confirmed_students: int = 1000,
    ) -> None:
        assert 0.0 <= confidence_threshold <= 1.0, "confidence_threshold must be between 0 and 1"
        assert max_confirmed_students > 0, "max_confirmed_students must be positive"

        self.confidence_threshold = confidence_threshold
        self.max_confirmed_students = max_confirmed_students

        self.confirmed_students: set[int] = set()
        self.current_active_ids: set[int] = set()

        self.total_unique_students = 0
        self.max_concurrent_students = 0

    def update_count(self, tracking_results: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(tracking_results, dict):
            raise TypeError("tracking_results must be a dictionary")

        active_ids = tracking_results.get("active_ids", set())
        track_confidences = tracking_results.get("track_confidences", {})

        if isinstance(active_ids, list):
            active_ids = set(active_ids)
        elif not isinstance(active_ids, set):
            active_ids = set()

        if not isinstance(track_confidences, dict):
            track_confidences = {}

        self.current_active_ids = active_ids

        confirmed_count = 0
        new_students = 0

        for track_id in active_ids:
            if not isinstance(track_id, (int, float)):
                continue

            confidence = track_confidences.get(track_id, 0.0)

            if confidence >= self.confidence_threshold:
                if track_id not in self.confirmed_students:
                    self.confirmed_students.add(track_id)
                    self.total_unique_students += 1
                    new_students += 1
                confirmed_count += 1

        self.max_concurrent_students = max(self.max_concurrent_students, len(active_ids))

        return {
            "current_count": len(active_ids),
            "confirmed_count": confirmed_count,
            "total_unique_students": self.total_unique_students,
            "max_concurrent_students": self.max_concurrent_students,
            "new_students": new_students,
            "active_ids": list(active_ids),
            "confirmed_student_ids": list(self.confirmed_students),
        }

    def reset(self) -> None:
        self.confirmed_students.clear()
        self.current_active_ids.clear()
        self.total_unique_students = 0
        self.max_concurrent_students = 0

    def get_statistics(self) -> dict[str, Any]:
        return {
            "current_count": len(self.current_active_ids),
            "total_unique_students": self.total_unique_students,
            "max_concurrent_students": self.max_concurrent_students,
            "confirmed_students": len(self.confirmed_students),
            "active_ids": list(self.current_active_ids),
            "max_confirmed_limit": self.max_confirmed_students,
        }
