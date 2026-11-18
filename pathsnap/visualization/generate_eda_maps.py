"""
pathsnap.visualization.generate_eda_maps
========================================
CLI script to generate representative interactive Folium EDA maps and screenshots
for PathSnap, saved into pathsnap/artifacts/maps/.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathsnap.visualization.map_visualizer import visualize_trip


def main():
    print("=" * 75)
    print("PATHSNAP FOLIUM GEOGRAPHICAL VISUALIZATION PIPELINE")
    print("=" * 75)

    output_dir = "pathsnap/artifacts/maps"
    os.makedirs(output_dir, exist_ok=True)

    trips_to_visualize = [
        ("Student folder/train/00000001", "00000001", "Training Set Trip 00000001 (Urban Center with Complex Network)"),
        ("Student folder/student_test/00000000", "00000000", "Student Test Trip 00000000 (Suburban Transit Corridor)"),
    ]

    results = []

    for trip_path, trip_id, desc in trips_to_visualize:
        print(f"\n>>> Generating Map for {desc}...")
        t0 = time.time()
        res = visualize_trip(
            trip_dir=trip_path,
            output_dir=output_dir,
            trip_id=trip_id,
            generate_screenshot=True,
        )
        elapsed = time.time() - t0
        res["elapsed_s"] = elapsed
        results.append(res)

        print(f"  Generated HTML Map:   {res['html_path']} ({os.path.getsize(res['html_path'])/1024:.1f} KB)")
        if res['screenshot_path']:
            print(f"  Captured Screenshot:  {res['screenshot_path']} ({os.path.getsize(res['screenshot_path'])/1024:.1f} KB)")
        else:
            print(f"  Screenshot status:    Skipped / Not captured")
        print(f"  GPS Observations:     {res['num_observations']:,}")
        print(f"  Candidate Arcs:       {res['total_candidates']:,}")
        print(f"  Ground Truth Route:   {'Included' if res['has_ground_truth'] else 'Not Present'}")
        print(f"  Render Latency:       {elapsed:.2f} s")

    print("\n" + "=" * 75)
    print("MAP GENERATION SUMMARY")
    print("=" * 75)
    for r in results:
        print(f"- Trip {r['trip_id']}:")
        print(f"    HTML:       {r['html_path']}")
        print(f"    Screenshot: {r['screenshot_path']}")
    print("\nAll interactive maps generated successfully inside 'pathsnap/artifacts/maps/'.")


if __name__ == "__main__":
    main()
