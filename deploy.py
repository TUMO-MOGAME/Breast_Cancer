#!/usr/bin/env python3
"""
Deployment helper script for the Breast Cancer Prediction App
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        'vercel.json',
        'best_svm_breast_cancer.joblib',
        'templates/index.html'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found!")
    return True

def test_local():
    """Test the application locally"""
    print("🧪 Testing model...")
    if not run_command("python test_model.py", "Model test"):
        return False
    
    print("🌐 Starting local server for testing...")
    print("   The server will start on http://localhost:5000")
    print("   Press Ctrl+C to stop the server when you're done testing")
    
    try:
        subprocess.run("python app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n✅ Local testing completed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Local server failed to start!")
        return False

def deploy_to_vercel():
    """Deploy to Vercel"""
    print("🚀 Deploying to Vercel...")
    
    # Check if Vercel CLI is installed
    try:
        subprocess.run("vercel --version", shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ Vercel CLI not found!")
        print("   Please install it with: npm install -g vercel")
        return False
    
    # Deploy
    if not run_command("vercel", "Vercel deployment"):
        return False
    
    print("🎉 Deployment completed!")
    print("   Your app should now be live on Vercel!")
    return True

def main():
    print("🏥 Breast Cancer Prediction App - Deployment Helper")
    print("=" * 60)
    print("Created by Tumo Olorato Mogame")
    print("=" * 60)
    
    if not check_requirements():
        sys.exit(1)
    
    print("\nWhat would you like to do?")
    print("1. Test locally")
    print("2. Deploy to Vercel")
    print("3. Both (test then deploy)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        test_local()
    elif choice == "2":
        deploy_to_vercel()
    elif choice == "3":
        print("🔄 Running full deployment process...")
        if test_local():
            input("\nPress Enter to continue with Vercel deployment...")
            deploy_to_vercel()
    else:
        print("❌ Invalid choice!")
        sys.exit(1)

if __name__ == "__main__":
    main()
