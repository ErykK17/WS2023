import pandas as pd

data = pd.read_csv('Electric_Vehicle_Population_Data.csv')
clean_data = data.dropna()
clean_data = clean_data.drop_duplicates()  # usunięcie wierszy z brakującymi danymi oraz duplikatów
valid_data = clean_data[
    (clean_data['Electric Range'] != 0)]  # usunięcie wierszy, gdzie zasięg na baterii auta jest niezbadany
valid_data = valid_data.copy()  # kopia dataframe'u aby uniknąć błędu SettingsWithCopyError
valid_data['Postal Code'] = pd.to_numeric(valid_data['Postal Code'], errors='coerce', downcast='integer').astype(
    'Int64')  # zamiana kodu pocztowego z float na int
print(valid_data.dtypes)  # wypisanie wszystkich typów danych by sprawdzić ich poprawność
valid_data['Valid Postal Code'] = valid_data['Postal Code'].astype(str).str.match(
    r'^\d{5}$')  # sprawdzenie czy kod pocztowy ma poprawny format (ma 5 cyfr)
