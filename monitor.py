import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import joblib


def monitor():
    reference = pd.DataFrame({
        'x':[1,2,3,4,5],
        'y':[2,4,6,8,10]
        
    })
    model = joblib.load('model.pkl')
    
    current_x = pd.DataFrame({'x':[7,8,13,9,11]})
    current_y = model.predict(current_x)
    
    current = pd.DataFrame({'x':current_x['x'],
                            'y':current_y})
    
    report = Report(metrics=[DataDriftPreset(drift_share=0.3)])
    report.run(reference_data = reference, current_data= current)
    report.save_html('drift_report.html')
    print("Drift report generated: drift_report.html")
    
if __name__ == '__main__':
    monitor()