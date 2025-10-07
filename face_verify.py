#!/usr/bin/env python3
import argparse
from deepface import DeepFace

def main():
    parser = argparse.ArgumentParser(
        description="Verify if two images are of the same person using DeepFace."
    )
    parser.add_argument("img1", help="Path to first image")
    parser.add_argument("img2", help="Path to second image")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="VGG-Face", 
        choices=["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace"],
        help="Face recognition model to use (default: VGG-Face)"
    )
    
    parser.add_argument(
        "--detector", 
        type=str, 
        default="opencv", 
        choices=["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"],
        help="Face detector backend to use (default: opencv)"
    )
    
    parser.add_argument(
        "--metric", 
        type=str, 
        default="cosine", 
        choices=["cosine", "euclidean", "euclidean_l2"],
        help="Distance metric to evaluate similarity (default: cosine)"
    )

    args = parser.parse_args()

    try:
        result = DeepFace.verify(
            img1_path=args.img1,
            img2_path=args.img2,
            model_name=args.model,
            detector_backend=args.detector,
            distance_metric=args.metric
        )

        is_verified = result["verified"]


        print("\n--- DeepFace Verification Result ---")
        if is_verified:
                print("✅ The two images are of the same person.")
        else:
                print("❌ The two images are NOT of the same person.")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Threshold: {result['threshold']:.4f}")
        print(f"Model: {args.model}")
        print(f"Detector: {args.detector}")
        print(f"Metric: {args.metric}")
        print("-----------------------------------\n")
 
    

  

    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    main()

