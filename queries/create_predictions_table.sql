CREATE TABLE IF NOT EXISTS lumendb.predictions_day (
    stay_date DATE,
    lower_bound INTEGER,
    upper_bound INTEGER,
    predicted_rooms INTEGER,
    total_rooms INTEGER
);

CREATE TABLE IF NOT EXISTS lumendb.predictions_week (
    stay_date DATE,
    lower_bound INTEGER,
    upper_bound INTEGER,
    predicted_rooms INTEGER,
    total_rooms INTEGER
);

CREATE TABLE IF NOT EXISTS lumendb.predictions_month (
    stay_date DATE,
    lower_bound INTEGER,
    upper_bound INTEGER,
    predicted_rooms INTEGER,
    total_rooms INTEGER
);