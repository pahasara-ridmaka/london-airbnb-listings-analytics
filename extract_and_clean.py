import pandas as pd

class ListingsExtractor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.keep_columns = [
            "neighbourhood_cleansed",
            "room_type",
            "property_type",
            "price",
            "accommodates",
            "bedrooms",
            "bathrooms",
            "amenities",
            "number_of_reviews",
            "reviews_per_month",
            "review_scores_rating",
            "availability_365",
        ]

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.input_path)

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.keep_columns].copy()
    
    def remove_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # convert price to int and remove price greater than 1000
        df["price"] = df["price"].str.replace("$", "").str.replace(",", "").astype(float)
        df = df[df["price"] <= 600]
        return df
    
    
    def remove_review_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # remove rows with number_of_reviews less than 5
        df = df[df["number_of_reviews"] >= 5]
        return df
    
    def remove_availability_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # remove rows with availability_365 less than 5
        df = df[df["availability_365"] >= 5]
        return df
    
    # def remove_minimum_nights_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
    #     # remove rows with minimum_nights greater than 7
    #     df = df[df["minimum_nights"] <= 7]
    #     return df
    
    
    def remove_bathrooms_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # remove rows with bathrooms greater than 3
        df = df[df["bathrooms"] <= 3]
        return df
    
    def remove_bedrooms_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # remove rows with bedrooms greater than 4
        df = df[df["bedrooms"] <= 4]

        # remove bedrooms 4 with price is less than 120
        df = df[~((df["bedrooms"] == 4) & (df["price"] < 120))]
        df = df[~((df['price'] >= 115) & (df['room_type'] == 'Private room'))]
        df = df[~((df['price'] >= 75) & (df['room_type'] == 'Shared room'))]
        df = df[~((df['price'] <=40) & (df['room_type'] == 'Entire home/apt'))]
        df = df[~((df['price'] >=330) & (df['room_type'] == 'Entire home/apt'))]
        df = df[~((df['price'] >=500) & (df['room_type'] == 'Hotel room'))]
        df = df[~((df['price'] >=40) & (df['room_type'] == 'Shared room'))]
        return df
    
    def remove_accommodates_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # remove rows with accommodates greater than 7
        df = df[df["accommodates"] <= 7]
        return df

    def convert_price_to_floating_point(self, df: pd.DataFrame) -> pd.DataFrame:
        df["price"] = df["price"].astype(float)
        return df
    
    def change_field_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={
            "neighbourhood_cleansed": "neighbourhood",
            "room_type": "room_type",
            "property_type": "property_type",
            "price": "price",
            "accommodates": "accommodates",
            "bedrooms": "bedrooms",
            "bathrooms": "bathrooms",
            "minimum_nights": "minimum_nights",
            "host_listings_count": "host_listings_count",
            "number_of_reviews": "number_of_reviews",
            "reviews_per_month": "reviews_per_month",
            "review_scores_rating": "review_scores_rating",
            "availability_365": "availability_365",
        })
        return df

    def save(self, df: pd.DataFrame) -> None:
        df.to_csv(self.output_path, index=False)

    def run(self) -> None:
        df = self.load()
        extracted = self.extract(df)
        no_empty = self.remove_empty_rows(extracted)
        no_outliers = self.remove_outliers(no_empty)
        no_review_outliers = self.remove_review_outliers(no_outliers)
        no_availability_outliers = self.remove_availability_outliers(no_review_outliers)
        # no_minimum_nights_outliers = self.remove_minimum_nights_outliers(no_availability_outliers)
        no_price_outliers = self.convert_price_to_floating_point(no_availability_outliers)
        no_bathrooms_outliers = self.remove_bathrooms_outliers(no_price_outliers)
        no_bedrooms_outliers = self.remove_bedrooms_outliers(no_bathrooms_outliers)
        no_accommodates_outliers = self.remove_accommodates_outliers(no_bedrooms_outliers)
        no_field_names = self.change_field_names(no_accommodates_outliers)
        self.save(no_field_names)

if __name__ == "__main__":
    extractor = ListingsExtractor(
        input_path="listings.csv",
        output_path="extracted_listings.csv",
    )
    extractor.run()