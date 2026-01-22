import os
import matplotlib.pyplot as plt


class Draw:
    def draw_boxes(self, frame, boxes):
        import cv2
        for x1, y1, x2, y2, score in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"person {score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (185, 255, 0),
                2
            )
        return frame

    def draw_metrics(self, video_name, metrics, save_dir):
        names = list(metrics.keys())
        values = list(metrics.values())

        plt.figure(figsize=(10, 5))
        bars = plt.bar(names, values)

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.title("Detection metrics")
        plt.xticks(rotation=20)
        plt.tight_layout()

        path = os.path.join(save_dir, f"{video_name.split('.')[0]}_metrics.png")
        plt.savefig(path)
        plt.close()
