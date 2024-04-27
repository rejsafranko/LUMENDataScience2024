CREATE TABLE IF NOT EXISTS lumendb.predictions (
    stay_date DATE,
    lower_bound INTEGER,
    upper_bound INTEGER,
    predicted_rooms INTEGER,
    total_rooms INTEGER
)