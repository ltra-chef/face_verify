#!/usr/bin/env python3
import logging
import os
import sys
# suppress general TensorFlow logging (including CUDA warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#disable oneDNN custom operations (if related warnings appear)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import argparse
from deepface import DeepFace
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def verify_faces(img1, img2, model, detector, metric):
    """Verify two images are of the same person"""
    result = DeepFace.verify(
        img1_path=img1,
        img2_path=img2,
        model_name=model,
        detector_backend=detector,
        distance_metric=metric
    )


    is_verified = result["verified"]


    print("\n--- DeepFace Verification Result ---")
    if is_verified:
        print("✅ The two images are of the same person.")
        print("Verified: True")        
    else:
        print("❌ The two images are NOT of the same person.")
        print("Verified: False")

    print(f"Distance: {result['distance']:.4f}")
    print(f"Threshold: {result['threshold']:.4f}")
    print(f"Model: {model}")
    print(f"Detector: {detector}")
    print(f"Metric: {metric}")
    print("-----------------------------------\n")
    



def find_faces(img, db_path, model, detector, metric, out_path="output.jpg"):
    """Find similar faces in a database folder and annotate the query image."""

    try:
        dfs = DeepFace.find(
            img_path=img,
            db_path=db_path,
            model_name=model,
            detector_backend=detector,
            distance_metric=metric,
            enforce_detection=True
        )
    except Exception as e:
        print(f"[!] Error running DeepFace.find: {e}", file=sys.stderr)
        return

    if isinstance(dfs, list) and len(dfs) > 0:
        df = dfs[0]
    else:
        df = dfs

    print("\n--- DeepFace Find Results ---")
    if df is None or df.empty:
        print("No matches found in database.")
        return

    match_details = []
    for df in dfs:
        match_detail = {}
        row = df.iloc[0]
        #print(row) # todo print this if verbose/debug enabled
        identity = row.identity
        person_name = os.path.basename(os.path.dirname(identity))
        match_detail["person_name"] = person_name
        match_detail["x"] = row.source_x
        match_detail["y"] = row.source_y
        match_detail["w"] = row.source_w
        match_detail["h"] = row.source_h
        distance = row.distance                                            
        confidence = row.confidence 
        
        print(f"Found: {person_name} at {identity} \t\t distance: {distance:.4f} \t confidence: {confidence:.4f}")
        match_details.append(match_detail)


    # Load query image
    image = cv2.imread(img)
    if image is None:
        print(f"Could not read image {img}")
        return

    # Get face bounding boxes in query image


    for match in match_details:
        x = match['x']
        y = match['y']
        w = match['w']
        h = match['h']
        person_name = match["person_name"]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, person_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save annotated image
    cv2.imwrite(out_path, image)
    print(f"Annotated image saved to {out_path}")

    # Optionally display inline (if running interactively, not in headless mode)
    try:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    except Exception:
        pass

def build_database_cache(db_path, model):
    print("Building the database, this may take a while...")
    # make image empty, and dont force a face detection
    DeepFace.find(img_path=np.ndarray(shape=[0,0,0]), enforce_detection=False,db_path=db_path, model_name=model)   
    print("...Done rebuilding the database!")


def main():
    parser = argparse.ArgumentParser(
        description="Face verification and search using DeepFace."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: verify
    verify_parser = subparsers.add_parser("verify", help="Verify if two images are the same person")
    verify_parser.add_argument("img1", help="Path to first image")
    verify_parser.add_argument("img2", help="Path to second image")

    # Subcommand: find
    find_parser = subparsers.add_parser("find", help="Find a face inside a database")
    find_parser.add_argument("img", help="Path to query image")
    find_parser.add_argument("db_path", help="Path to database folder containing face images")
    find_parser.add_argument("--output", type=str, default="out.jpg", help="Path write output file to")

    # Subcommand: build
    build_parser = subparsers.add_parser("build", help="Rebuild the database cache")
    build_parser.add_argument("db_path", help="Path to database folder containing face images")

    # Shared arguments
    for sub in [verify_parser, find_parser, build_parser]:
        sub.add_argument(
            "--model", 
            type=str, 
            default="VGG-Face", 
            choices=["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace"],
            help="Face recognition model (default: VGG-Face)"
        )
        sub.add_argument(
            "--detector", 
            type=str, 
            default="opencv", 
            choices=["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"],
            help="Face detector backend (default: opencv)"
        )
        sub.add_argument(
            "--metric", 
            type=str, 
            default="cosine", 
            choices=["cosine", "euclidean", "euclidean_l2"],
            help="Distance metric (default: cosine)"
        )

    args = parser.parse_args()

    if args.command == "verify":
        verify_faces(args.img1, args.img2, args.model, args.detector, args.metric)

    elif args.command == "find":
        find_faces(args.img, args.db_path, args.model, args.detector, args.metric,args.output)

    elif args.command == "build":
        build_database_cache(args.db_path, args.model)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
