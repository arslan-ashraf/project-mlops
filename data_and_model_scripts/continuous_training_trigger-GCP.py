# requirements.txt
# google-cloud-aiplatform[pipelines]

import functions_framework
from google.cloud import aiplatform

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def continuous_training(cloud_event):
    data = cloud_event.data

    # bucket the trigger is set to go off on
    data_bucket_name = data["bucket"]
    DATA_BUCKET_URI = "gs://" + data_bucket_name

    print("=" * 100)
    print("DATA_BUCKET_URI:", DATA_BUCKET_URI)

    fine_tuning_parameter_values = { 
        "first_time_training": False,
        "data_bucket_uri": DATA_BUCKET_URI,
        "processed_data_save_bucket_uri": PROCESSED_DATA_SAVE_BUCKET_URI,
        "new_train_data_bucket_uri": PROCESSED_DATA_SAVE_BUCKET_URI,
        "valid_data_bucket_uri": VALID_DATA_BUCKET_URI,
        "fraction_for_valid_and_test_data": "0.0",
        "mae_threshold": 5.0,
        "fine_tuning_epochs": "10",
        "fine_tuning_learning_rate": "0.0005"
    }

    print("fine tuning parameters:")
    print(fine_tuning_parameter_values)
    print("=" * 100)
    
    job = aiplatform.PipelineJob(
        display_name="used_cars_kubeflow_pipeline_training_job",
        template_path=PIPELINE_YAML_FILE_URI,
        pipeline_root=PIPELINE_BUCKET_URI,
        enable_caching=False,
        parameter_values=fine_tuning_parameter_values
    )

    print("Starting continuous training job ...")
                                        
    job.run()