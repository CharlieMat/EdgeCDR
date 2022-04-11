cd {root_path}public/amz_rating/ # root_path must be the same as ROOT_PATH in preprocess.py
mkdir datasets

# Amazon data
mkdir datasets/amz_rating
cd datasets/amz_rating

wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/All_Beauty.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Appliances.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Arts_Crafts_and_Sewing.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Automotive.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Books.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/CDs_and_Vinyl.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Cell_Phones_and_Accessories.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Clothing_Shoes_and_Jewelry.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Digital_Music.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Electronics.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Gift_Cards.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Grocery_and_Gourmet_Food.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Home_and_Kitchen.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Industrial_and_Scientific.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Kindle_Store.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Luxury_Beauty.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Magazine_Subscriptions.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Movies_and_TV.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Musical_Instruments.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Office_Products.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Patio_Lawn_and_Garden.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Pet_Supplies.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Prime_Pantry.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Software.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Sports_and_Outdoors.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Tools_and_Home_Improvement.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Toys_and_Games.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games.csv

