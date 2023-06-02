import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from taxifare.params import *
from taxifare.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from taxifare.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from taxifare.ml_logic.preprocessor import preprocess_features
from taxifare.ml_logic.registry import load_model, save_model, save_results


def preprocess(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Query raw data from BigQuery using `get_data_with_cache`
    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM {GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
    """

    # YOUR CODE HERE✅

    # Retrieve `query` data from BigQuery or from `data_query_cache_path` if the file already exists!
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_query_cached_exists = data_query_cache_path.is_file()

    if data_query_cached_exists:
        print("Loading data from local CSV...")

        data = pd.read_csv(data_query_cache_path)

    else:
        print("Loading data from Querying Big Query server...")

        client = bigquery.Client(project=GCP_PROJECT)
        query_job = client.query(query)
        result = query_job.result()
        data = result.to_dataframe()

        # Save it locally to accelerate the next queries!
        data.to_csv(data_query_cache_path, header=True, index=False)

    # Clean data using data.py
    data = clean_data(data)

    # Create (X_train, y_train, X_val, y_val) without data leaks
    # No need for test sets, we'll report val metrics only
    split_ratio = 0.02 # About one month of validation data

    train_length = int(len(data) * (1 - split_ratio))

    data_train = data.iloc[:train_length, :].sample(frac=1)
    data_val = data.iloc[train_length:, :].sample(frac=1)

    X_train = data_train.drop("fare_amount", axis=1)
    y_train = data_train[["fare_amount"]]

    X_val = data_val.drop("fare_amount", axis=1)
    y_val = data_val[["fare_amount"]]

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()
    # YOUR CODE HERE✅
    load_data_to_bq(X_train, table_name=f"{GCP_PROJECT_WAGON}.{BQ_DATASET}.processed_data")

    print("✅ preprocess() done \n")


def train(
        min_date:str = '2009-01-01',
        max_date:str = '2015-01-01',
        split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.0005,
        batch_size = 256,
        patience = 2
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!

    # YOUR CODE HERE✅
    query = f"""
        SELECT *
        FROM {GCP_PROJECT_WAGON}.{BQ_DATASET}.processed_data
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
    """
    processed_data = get_data_with_cache(query=query, filename="processed_data.csv")

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    # YOUR CODE HERE✅
    X_train_processed = preprocess_features(X_train)
    X_val_processed = preprocess_features(X_val)


    ##################################################

    # Train model using `model.py`
    # YOUR CODE HERE✅

    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    model = None
    metrics_val_list = []  # store the val_mae of each chunk

    # Iterate in chunks and partially fit on each chunk
    chunks = pd.read_csv(
        data_processed_path,
        chunksize=CHUNK_SIZE,
        header=None,
        dtype=DTYPES_PROCESSED
    )

    for chunk_id, chunk in enumerate(chunks):
        print(f"Training on preprocessed chunk n°{chunk_id}")

        # You can adjust training params for each chunk if you want!
        learning_rate = 0.0005
        batch_size = 256
        patience=2
        split_ratio = 0.1 # Higher train/val split ratio when chunks are small! Feel free to adjust.

        # Create (X_train_chunk, y_train_chunk, X_val_chunk, y_val_chunk)
        train_length = int(len(chunk)*(1-split_ratio))
        chunk_train = chunk.iloc[:train_length, :].sample(frac=1).to_numpy()
        chunk_val = chunk.iloc[train_length:, :].sample(frac=1).to_numpy()

        X_train_chunk = chunk_train[:, :-1]
        y_train_chunk = chunk_train[:, -1]
        X_val_chunk = chunk_val[:, :-1]
        y_val_chunk = chunk_val[:, -1]

        # Train a model *incrementally*, and store the val_mae of each chunk in `metrics_val_list`
        if model is None:
            model = initialize_model(input_shape=X_train_chunk.shape[1:])

        model = compile_model(model, learning_rate)

        model, history = train_model(
            model,
            X_train_chunk,
            y_train_chunk,
            batch_size=batch_size,
            patience=patience,
            validation_data=(X_val_chunk, y_val_chunk)
        )

        metrics_val_chunk = np.min(history.history['val_mae'])
        metrics_val_list.append(metrics_val_chunk)


    # Return the last value of the validation MAE
    val_mae = metrics_val_list[-1]

    # Save model and training params
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        incremental=True,
        chunk_size=CHUNK_SIZE
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return val_mae


def evaluate(
        min_date:str = '2014-01-01',
        max_date:str = '2015-01-01',
        stage: str = "Production"
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage)
    assert model is not None

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    pass  # YOUR CODE HERE

    if data_processed.shape[0] == 0:
        print("❌ No data to evaluate on")
        return None

    data_processed = data_processed.to_numpy()

    X_new = data_processed[:, :-1]
    y_new = data_processed[:, -1]

    metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    mae = metrics_dict["mae"]

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=DATA_SIZE,
        row_count=len(X_new)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
        pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
        pickup_longitude=[-73.950655],
        pickup_latitude=[40.783282],
        dropoff_longitude=[-73.984365],
        dropoff_latitude=[40.769802],
        passenger_count=[1],
    ))

    model = load_model()
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred


if __name__ == '__main__':
    preprocess(min_date='2009-01-01', max_date='2015-01-01')
    train(min_date='2009-01-01', max_date='2015-01-01')
    evaluate(min_date='2009-01-01', max_date='2015-01-01')
    pred()
