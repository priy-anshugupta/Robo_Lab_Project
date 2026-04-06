"""
ISL-Multilingual Bridge — Data Collector
Records hand landmark data from webcam for each ISL sign label.
Creates CSV datasets for training the sign classifiers.

Usage:
    python training/data_collector.py --label HELLO --samples 200
    python training/data_collector.py --label A --samples 200 --mode static
"""

import os
import sys
import csv
import cv2
import numpy as np
import argparse
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DATA_DIR, SINGLE_HAND_FEATURES
from vision.landmark_extractor import LandmarkExtractor

logger = logging.getLogger("isl-bridge.training.collector")


class DataCollector:
    """
    Collects hand landmark data from webcam for training.
    Records normalized feature vectors with sign labels to CSV files.
    """

    def __init__(self, output_dir: Path = RAW_DATA_DIR):
        """
        Args:
            output_dir: Directory to save CSV data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = LandmarkExtractor(static_image_mode=False)

    def collect_static(
        self,
        label: str,
        num_samples: int = 200,
        camera_id: int = 0,
        delay_ms: int = 100,
    ):
        """
        Collect static sign data (single-frame landmarks).

        Args:
            label: Sign label (e.g., "HELLO", "A")
            num_samples: Number of samples to collect
            camera_id: Camera index
            delay_ms: Delay between captures in milliseconds
        """
        label = label.upper()
        output_file = self.output_dir / f"{label}_static.csv"

        print(f"\n{'='*60}")
        print(f"  ISL Data Collector — Static Sign: {label}")
        print(f"  Target: {num_samples} samples → {output_file}")
        print(f"{'='*60}")
        print(f"\n  Instructions:")
        print(f"  1. Show the sign for '{label}' to the camera")
        print(f"  2. Press 'S' to START recording")
        print(f"  3. Press 'Q' to QUIT at any time")
        print(f"  4. Keep your hand visible and steady")
        print()

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("ERROR: Cannot open camera!")
            return

        # Prepare CSV header
        headers = [f"lm_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
        headers.append("label")

        samples_collected = 0
        recording = False
        data_rows = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks = self.extractor.extract(frame)
            display_frame = self.extractor.draw_landmarks(frame, landmarks)

            # Status display
            status = "RECORDING" if recording else "PAUSED (Press S to start)"
            color = (0, 0, 255) if recording else (0, 165, 255)
            cv2.putText(display_frame, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Status: {status}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_frame, f"Samples: {samples_collected}/{num_samples}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Progress bar
            progress = int(500 * samples_collected / num_samples)
            cv2.rectangle(display_frame, (10, 100), (510, 120), (50, 50, 50), -1)
            cv2.rectangle(display_frame, (10, 100), (10 + progress, 120), (0, 255, 0), -1)

            # Record data if recording and hand detected
            if recording and landmarks["has_hands"] and landmarks["features"] is not None:
                row = landmarks["features"].tolist()
                row.append(label)
                data_rows.append(row)
                samples_collected += 1

                if samples_collected >= num_samples:
                    print(f"\n✅ Collected {num_samples} samples for '{label}'!")
                    break

                cv2.waitKey(delay_ms)

            cv2.imshow("ISL Data Collector", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n⏹ Collection stopped by user.")
                break
            elif key == ord("s"):
                recording = not recording
                state = "Started" if recording else "Paused"
                print(f"  ⏺ Recording {state}")

        # Save collected data
        if data_rows:
            file_exists = output_file.exists()
            with open(output_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(headers)
                writer.writerows(data_rows)

            print(f"\n💾 Saved {len(data_rows)} samples to {output_file}")
        else:
            print("\n⚠ No data collected.")

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()

    def collect_dynamic(
        self,
        label: str,
        num_sequences: int = 50,
        sequence_length: int = 30,
        camera_id: int = 0,
    ):
        """
        Collect dynamic sign data (multi-frame sequences).

        Args:
            label: Sign label
            num_sequences: Number of 30-frame sequences to collect
            sequence_length: Frames per sequence
            camera_id: Camera index
        """
        label = label.upper()
        output_file = self.output_dir / f"{label}_dynamic.npy"

        print(f"\n{'='*60}")
        print(f"  ISL Data Collector — Dynamic Sign: {label}")
        print(f"  Target: {num_sequences} sequences of {sequence_length} frames")
        print(f"{'='*60}")
        print(f"\n  Press 'S' to start recording a sequence")
        print(f"  Press 'Q' to quit\n")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("ERROR: Cannot open camera!")
            return

        sequences = []
        seq_count = 0

        while seq_count < num_sequences:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks = self.extractor.extract(frame)
            display_frame = self.extractor.draw_landmarks(frame, landmarks)

            cv2.putText(display_frame, f"Dynamic: {label} | Seq: {seq_count}/{num_sequences}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'S' to record sequence",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.imshow("ISL Data Collector", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Record a sequence
                sequence = []
                print(f"  ⏺ Recording sequence {seq_count + 1}...")

                for frame_idx in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    landmarks = self.extractor.extract(frame)
                    display_frame = self.extractor.draw_landmarks(frame, landmarks)

                    cv2.putText(display_frame,
                                f"Recording: {frame_idx + 1}/{sequence_length}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("ISL Data Collector", display_frame)
                    cv2.waitKey(33)  # ~30fps

                    if landmarks["has_hands"] and landmarks["features"] is not None:
                        sequence.append(landmarks["features"])
                    else:
                        sequence.append(np.zeros(SINGLE_HAND_FEATURES))

                if len(sequence) == sequence_length:
                    sequences.append(np.array(sequence))
                    seq_count += 1
                    print(f"  ✅ Sequence {seq_count} recorded")

        # Save sequences
        if sequences:
            all_sequences = np.array(sequences)  # (N, 30, 63)
            np.save(output_file, all_sequences)
            print(f"\n💾 Saved {len(sequences)} sequences to {output_file}")
        else:
            print("\n⚠ No sequences collected.")

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()


def main():
    parser = argparse.ArgumentParser(description="ISL Sign Language Data Collector")
    parser.add_argument("--label", type=str, required=True, help="Sign label to collect")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--mode", type=str, default="static", choices=["static", "dynamic"],
                        help="Collection mode")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")

    args = parser.parse_args()

    collector = DataCollector()

    if args.mode == "static":
        collector.collect_static(args.label, args.samples, args.camera)
    else:
        collector.collect_dynamic(args.label, args.samples, camera_id=args.camera)


if __name__ == "__main__":
    main()
