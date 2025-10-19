"""
Student Counter Module

This module provides the StudentCounter class for counting and tracking student
statistics across video frames. It maintains counts of active students, unique
students seen over time, and maximum concurrent students.

The counter uses confidence thresholds to filter which tracks should be counted
as confirmed students, helping to reduce false positives from the detection/tracking
pipeline.

Example:
    >>> counter = StudentCounter(confidence_threshold=0.5)
    >>> counts = counter.update_count(tracking_results)
    >>> print(f"Current students: {counts['current_count']}")
"""

from typing import Any


class StudentCounter:
    """Student counting and statistics tracker.
    
    This class maintains student count statistics across video frames. It tracks:
    - Current number of active students in the frame
    - Total unique students seen throughout the video
    - Maximum concurrent students at any point
    - Confirmed student IDs based on confidence threshold
    
    The counter uses a confidence threshold to filter tracks, ensuring only
    high-confidence detections are counted as confirmed students.
    
    Attributes:
        confidence_threshold (float): Minimum confidence to count as student (0.0-1.0)
        max_confirmed_students (int): Maximum number of students to track
        confirmed_students (set[int]): Set of confirmed student track IDs
        current_active_ids (set[int]): Currently active track IDs
        total_unique_students (int): Total unique students seen
        max_concurrent_students (int): Maximum students seen simultaneously
    
    Example:
        >>> counter = StudentCounter(
        ...     confidence_threshold=0.5,
        ...     max_confirmed_students=1000
        ... )
        >>> counts = counter.update_count(tracking_results)
        >>> print(f"Current: {counts['current_count']}")
        >>> print(f"Total unique: {counts['total_unique_students']}")
    """
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        max_confirmed_students: int = 1000,
    ) -> None:
        """Initialize student counter with configuration.
        
        Creates a new counter instance with specified thresholds and limits.
        The counter starts with empty statistics and builds them up as frames
        are processed.
        
        Args:
            confidence_threshold (float): Minimum confidence to count as student (0.0-1.0).
                                        Higher values = stricter counting. Defaults to 0.5.
            max_confirmed_students (int): Maximum number of students to track.
                                        Defaults to 1000.
        
        Raises:
            AssertionError: If confidence_threshold is out of range or max_confirmed_students <= 0.
        
        Example:
            >>> counter = StudentCounter(
            ...     confidence_threshold=0.5,
            ...     max_confirmed_students=1000
            ... )
        """
        # Validate parameters
        assert 0.0 <= confidence_threshold <= 1.0, "confidence_threshold must be between 0 and 1"
        assert max_confirmed_students > 0, "max_confirmed_students must be positive"

        # Store configuration
        self.confidence_threshold = confidence_threshold
        self.max_confirmed_students = max_confirmed_students

        # Initialize tracking sets
        self.confirmed_students: set[int] = set()  # All confirmed student IDs
        self.current_active_ids: set[int] = set()  # Currently active IDs

        # Initialize statistics
        self.total_unique_students = 0
        self.max_concurrent_students = 0

    def update_count(self, tracking_results: dict[str, Any]) -> dict[str, Any]:
        """Update student counts based on tracking results.
        
        This method processes tracking results from the tracker and updates
        all student count statistics. It filters tracks by confidence threshold
        to determine which should be counted as confirmed students.
        
        The method:
        1. Extracts active track IDs from tracking results
        2. Filters tracks by confidence threshold
        3. Updates confirmed students set
        4. Updates statistics (total unique, max concurrent)
        5. Returns current count information
        
        Args:
            tracking_results (dict[str, Any]): Tracking results from tracker containing:
                - "active_ids": Set/list of currently active track IDs
                - "track_confidences": Dict mapping track IDs to confidence scores
        
        Returns:
            dict[str, Any]: Dictionary containing:
                - "current_count": Number of currently active students
                - "confirmed_count": Number of confirmed students (above threshold)
                - "total_unique_students": Total unique students seen
                - "max_concurrent_students": Maximum students seen at once
                - "new_students": Number of new students in this frame
                - "active_ids": List of active track IDs
                - "confirmed_student_ids": List of confirmed student IDs
        
        Raises:
            TypeError: If tracking_results is not a dictionary.
        
        Example:
            >>> counts = counter.update_count(tracking_results)
            >>> print(f"Current: {counts['current_count']}")
            >>> print(f"Total unique: {counts['total_unique_students']}")
        """
        # Validate input
        if not isinstance(tracking_results, dict):
            raise TypeError("tracking_results must be a dictionary")

        # Extract active track IDs and confidences
        active_ids = tracking_results.get("active_ids", set())
        track_confidences = tracking_results.get("track_confidences", {})

        # Convert active_ids to set if it's a list
        if isinstance(active_ids, list):
            active_ids = set(active_ids)
        elif not isinstance(active_ids, set):
            active_ids = set()

        # Ensure track_confidences is a dict
        if not isinstance(track_confidences, dict):
            track_confidences = {}

        # Update current active IDs
        self.current_active_ids = active_ids

        # Initialize counters
        confirmed_count = 0  # Students above confidence threshold
        new_students = 0  # New students in this frame

        # Process each active track
        for track_id in active_ids:
            # Skip invalid track IDs
            if not isinstance(track_id, (int, float)):
                continue

            # Get confidence for this track
            confidence = track_confidences.get(track_id, 0.0)

            # Count as confirmed student if above threshold
            if confidence >= self.confidence_threshold:
                # Check if this is a new student
                if track_id not in self.confirmed_students:
                    self.confirmed_students.add(track_id)
                    self.total_unique_students += 1
                    new_students += 1
                confirmed_count += 1

        # Update maximum concurrent students
        self.max_concurrent_students = max(self.max_concurrent_students, len(active_ids))

        # Return comprehensive count information
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
        """Reset all counters and statistics to initial state.
        
        Clears all tracked student IDs and resets statistics to zero.
        Useful for restarting tracking during a session without stopping
        the application.
        
        Returns:
            None
        
        Example:
            >>> counter.reset()
            >>> # All statistics are now zero
        """
        self.confirmed_students.clear()
        self.current_active_ids.clear()
        self.total_unique_students = 0
        self.max_concurrent_students = 0

    def get_statistics(self) -> dict[str, Any]:
        """Get current statistics without updating counts.
        
        Returns a snapshot of current statistics without processing new
        tracking results. Useful for querying statistics at any time.
        
        Returns:
            dict[str, Any]: Dictionary containing:
                - "current_count": Number of currently active students
                - "total_unique_students": Total unique students seen
                - "max_concurrent_students": Maximum students at once
                - "confirmed_students": Number of confirmed students
                - "active_ids": List of active track IDs
                - "max_confirmed_limit": Maximum allowed students
        
        Example:
            >>> stats = counter.get_statistics()
            >>> print(f"Total unique: {stats['total_unique_students']}")
        """
        return {
            "current_count": len(self.current_active_ids),
            "total_unique_students": self.total_unique_students,
            "max_concurrent_students": self.max_concurrent_students,
            "confirmed_students": len(self.confirmed_students),
            "active_ids": list(self.current_active_ids),
            "max_confirmed_limit": self.max_confirmed_students,
        }
