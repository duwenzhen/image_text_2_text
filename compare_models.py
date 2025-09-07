#!/usr/bin/env python3
"""
Compare Gemma-3-4B-IT and Qwen2.5-VL-7B-Instruct models for image description with metadata.
"""

import os
import sys
import time
from datetime import datetime

def run_gemma():
    """Run the Gemma model test."""
    print("🔄 Running Gemma-3-4B-IT model...")
    print("="*80)
    
    try:
        # Import and run the gemma test
        import subprocess
        result = subprocess.run(
            ['poetry', 'run', 'python', 'test_gemma.py'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode == 0:
            print("✅ Gemma model completed successfully")
            print("\nGEMMA OUTPUT:")
            print("-" * 40)
            print(result.stdout)
            print("-" * 40)
        else:
            print("❌ Gemma model failed")
            print("Error:", result.stderr)
            
        return result.stdout if result.returncode == 0 else None
        
    except Exception as e:
        print(f"❌ Error running Gemma: {e}")
        return None

def run_qwen():
    """Run the Qwen model test."""
    print("\n🔄 Running Qwen2.5-VL-7B-Instruct model...")
    print("="*80)
    
    try:
        # Import and run the qwen test
        import subprocess
        result = subprocess.run(
            ['poetry', 'run', 'python', 'test_qwen.py'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode == 0:
            print("✅ Qwen model completed successfully")
            print("\nQWEN OUTPUT:")
            print("-" * 40)
            print(result.stdout)
            print("-" * 40)
        else:
            print("❌ Qwen model failed")
            print("Error:", result.stderr)
            
        return result.stdout if result.returncode == 0 else None
        
    except Exception as e:
        print(f"❌ Error running Qwen: {e}")
        return None

def save_comparison(gemma_output, qwen_output):
    """Save the comparison results to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./data/model_comparison_{timestamp}.txt"
    
    try:
        with open(filename, 'w') as f:
            f.write("MODEL COMPARISON RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image: IMG_1854.jpg\n")
            f.write("="*80 + "\n\n")
            
            f.write("GEMMA-3-4B-IT RESPONSE:\n")
            f.write("-"*40 + "\n")
            if gemma_output:
                f.write(gemma_output)
            else:
                f.write("Failed to generate response\n")
            f.write("\n" + "-"*40 + "\n\n")
            
            f.write("QWEN2.5-VL-7B-INSTRUCT RESPONSE:\n")
            f.write("-"*40 + "\n")
            if qwen_output:
                f.write(qwen_output)
            else:
                f.write("Failed to generate response\n")
            f.write("\n" + "-"*40 + "\n\n")
            
            f.write("COMPARISON NOTES:\n")
            f.write("-"*20 + "\n")
            f.write("• Both models received the same metadata-enhanced prompt\n")
            f.write("• Both models were asked to create natural, story-like descriptions\n")
            f.write("• Metadata included: location (UAE), time (Sept 7, 2025 7:05 AM), camera (iPhone 15 Pro)\n")
        
        print(f"\n💾 Comparison saved to: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Error saving comparison: {e}")
        return None

def main():
    """Main comparison function."""
    print("🚀 VISION MODEL COMPARISON")
    print("="*80)
    print("Comparing Gemma-3-4B-IT vs Qwen2.5-VL-7B-Instruct")
    print("Image: IMG_1854.jpg (with metadata)")
    print("="*80)
    
    # Check if image exists
    image_path = "./data/IMG_1854.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        print("Please make sure the image file exists before running the comparison.")
        return
    
    start_time = time.time()
    
    # Run both models
    gemma_result = run_gemma()
    qwen_result = run_qwen()
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total execution time: {total_time:.1f} seconds")
    
    # Save comparison
    if gemma_result or qwen_result:
        comparison_file = save_comparison(gemma_result, qwen_result)
        
        # Print summary
        print("\n📊 COMPARISON SUMMARY:")
        print("="*50)
        
        if gemma_result:
            print("✅ Gemma-3-4B-IT: Successfully generated description")
        else:
            print("❌ Gemma-3-4B-IT: Failed to generate description")
            
        if qwen_result:
            print("✅ Qwen2.5-VL-7B-Instruct: Successfully generated description")
        else:
            print("❌ Qwen2.5-VL-7B-Instruct: Failed to generate description")
        
        if comparison_file:
            print(f"\n📁 Full comparison saved to: {comparison_file}")
    else:
        print("\n❌ Both models failed - no comparison to save")

if __name__ == "__main__":
    main()