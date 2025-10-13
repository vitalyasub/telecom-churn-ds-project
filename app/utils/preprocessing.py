import pandas as pd

def prepare_input_data(
    is_tv_subscriber,
    is_movie_package_subscriber,
    subscription_age,
    bill_avg,
    reamining_contract,
    service_failure_count,
    download_avg,
    upload_avg,
    download_over_limit
):
    data = {
        "is_tv_subscriber": 1 if is_tv_subscriber == "Так" else 0,
        "is_movie_package_subscriber": 1 if is_movie_package_subscriber == "Так" else 0,
        "subscription_age": subscription_age,
        "bill_avg": bill_avg,
        "reamining_contract": reamining_contract,
        "service_failure_count": service_failure_count,
        "download_avg": download_avg,
        "upload_avg": upload_avg,
        "download_over_limit": 1 if download_over_limit == "Так" else 0
    }
    return pd.DataFrame([data])