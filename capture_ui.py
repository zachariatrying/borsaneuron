import os
import sys
import time
import subprocess
from playwright.sync_api import sync_playwright

def main():
    workspace_dir = r"C:\Users\ibrah\.gemini\antigravity\scratch\ipo_analyzer"
    os.chdir(workspace_dir)
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    out_dir = os.path.join(workspace_dir, "sunum_gorselleri")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Start Streamlit server in the background
    print("Starting Streamlit server on port 8501...")
    cmd = [sys.executable, "-m", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.headless=true"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    
    # Wait for the server to start
    print("Waiting 10 seconds for Streamlit to start...")
    time.sleep(10)
    
    # 2. Run Playwright to capture screenshots of each page
    print("Launching Playwright...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()
            
            print("Navigating to Streamlit app at http://localhost:8501...")
            page.goto("http://localhost:8501", timeout=30000)
            
            # Wait for content to load
            page.wait_for_selector(".stApp", timeout=15000)
            print("App loaded successfully!")
            time.sleep(3)
            
            # --- PAGE 1: Welcome page (System Overview & Documentation) ---
            print("Capturing Welcome page...")
            path_welcome = os.path.join(out_dir, "03_veri_kesfi.png")
            page.screenshot(path=path_welcome)
            print(f"Saved Welcome screenshot to: {path_welcome}")
            
            # Helper selector to click sidebar radio options robustly
            def navigate_to(page_text):
                print(f"Navigating to sidebar page: {page_text}...")
                selector = f"text={page_text}"
                page.click(selector)
                time.sleep(3) # Wait for page state to change and render
            
            # --- PAGE 2: Exploratory Data Analysis (EDA) ---
            navigate_to("Exploratory Data Analysis (EDA)")
            path1 = os.path.join(out_dir, "borsaneuron_ui_dashboard.png")
            page.screenshot(path=path1)
            print(f"Saved EDA page screenshot to: {path1}")
            
            # --- PAGE 3: Feature Correlation & Selection ---
            navigate_to("Feature Correlation & Selection")
            path2 = os.path.join(out_dir, "04_korelasyon_heatmap.png")
            page.screenshot(path=path2)
            print(f"Saved Correlation page screenshot to: {path2}")
            
            # --- PAGE 4: Market Regime Clustering ---
            navigate_to("Market Regime Clustering")
            path3 = os.path.join(out_dir, "05_kmeans_pca.png")
            page.screenshot(path=path3)
            print(f"Saved Market Regimes page screenshot to: {path3}")
            
            # --- PAGE 5: Machine Learning Model Analysis ---
            navigate_to("Machine Learning Model Analysis")
            train_btn = page.locator("button:has-text('Start Model Training Matrix')")
            if train_btn.count() > 0:
                print("Clicking 'Start Model Training Matrix' button...")
                train_btn.click()
                print("Waiting for model training to complete (15s)...")
                time.sleep(15)
            path4 = os.path.join(out_dir, "model_4_karsilastirma.png")
            page.screenshot(path=path4)
            print(f"Saved Model Comparison page screenshot to: {path4}")
            
            # --- PAGE 6: Time-Series Trend Forecasting ---
            navigate_to("Time-Series Trend Forecasting")
            tahmin_btn = page.locator("button:has-text('Run Forecast Matrix')")
            if tahmin_btn.count() > 0:
                print("Clicking 'Run Forecast Matrix' button...")
                tahmin_btn.click()
                print("Waiting for Prophet modeling (10s)...")
                time.sleep(10)
            path5 = os.path.join(out_dir, "prophet_forecast_real.png")
            page.screenshot(path=path5)
            print(f"Saved Prophet page screenshot to: {path5}")
            
            # --- PAGE 7: Live Stock Query & Inference ---
            navigate_to("Live Stock Query & Inference")
            analiz_btn = page.locator("button:has-text('Analyze Stock')")
            if analiz_btn.count() > 0:
                print("Clicking 'Analyze Stock' button...")
                analiz_btn.click()
                print("Waiting for live data fetch & inference (6s)...")
                time.sleep(6)
            path7 = os.path.join(out_dir, "borsaneuron_hisse_sorgu_real.png")
            page.screenshot(path=path7)
            print(f"Saved Live Stock Query page screenshot to: {path7}")
            
            # --- PAGE 8: Portfolio Backtesting & Simulation ---
            navigate_to("Portfolio Backtesting & Simulation")
            backtest_btn = page.locator("button:has-text('Run Out-of-Sample Backtest')")
            if backtest_btn.count() > 0:
                print("Clicking 'Run Out-of-Sample Backtest' button...")
                backtest_btn.click()
                print("Waiting for Backtest run (10s)...")
                time.sleep(10)
            path6 = os.path.join(out_dir, "borsaneuron_scenario_ui.png")
            page.screenshot(path=path6)
            print(f"Saved Backtest page screenshot to: {path6}")
            
            # --- PAGE 9: Automated Pattern Scanner ---
            navigate_to("Automated Pattern Scanner")
            scan_btn = page.locator("button:has-text('Start Live Scan')")
            if scan_btn.count() > 0:
                print("Clicking 'Start Live Scan' button...")
                scan_btn.click()
                print("Waiting for Live Scan (10s)...")
                time.sleep(10)
            path8 = os.path.join(out_dir, "senaryo_kume_profil.png") # Override target placeholder or new one
            page.screenshot(path=path8)
            print(f"Saved Pattern Scanner page screenshot to: {path8}")
            
            browser.close()
            print("[SUCCESS] All corporate layout screenshots taken successfully!")
            
    except Exception as e:
        print(f"[ERROR] Playwright execution failed: {e}")
    finally:
        # Terminate Streamlit server
        print("Terminating Streamlit server process...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print("Streamlit process terminated.")
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            print("Streamlit process killed.")

if __name__ == "__main__":
    main()
