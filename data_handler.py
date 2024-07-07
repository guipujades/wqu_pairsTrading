import zipfile
import requests
import tempfile
import pandas as pd
from glob import glob
from pathlib import Path

from data_utils import complete_b3_fiis


class Tickers_B3():
    """
    Automatically captures tickers of assets from B3 from the website,
    with identification of company name, sector, and subsector.
    """

    def __init__(self, tmp_path):
        """
        Initializes the Tickers_B3 class with a temp path for file storage.

        :param tmp_path: Temp file path for storing downloaded files.
        """
        self.tmp_path = tmp_path
        
        home = Path.home()
        database_path = Path(home, 'Documents', 'GitHub', 'database')
        self.assets_path = Path(database_path, 'Codigos Ativos B3.xlsx') # file for data filtering
        
    def add_11(self, item):
        """
        Data adaptation to avoid including FIIs.

        :param item: The item to be modified.
        :return: The modified item with '11' appended.
        """
        return str(item) + '11'

    def open_file(self, local_doc=True):
        """
        Downloads a zip file from B3 containing sector information and extracts it.
        """
        
        if local_doc:
            pass
            
        else: 
            url = 'https://bvmf.bmfbovespa.com.br/InstDados/InformacoesEmpresas/ClassifSetorial.zip'
            req = requests.get(url)
            
            # Split for file name
            filename = url.split('/')[-1]
            
            # Temp file
            with open(self.tmp_path + '/' + filename,'wb') as output_file:
                output_file.write(req.content)
            
            with zipfile.ZipFile(self.tmp_path + '/' + filename,'r') as zip_ref:
                zip_ref.extractall(self.tmp_path)

    def get_tickers(self, yfinance=True, local_doc=True):
        """
        Opens the downloaded xlsx file from B3, organizes data,
        saves the dataframe, and returns a list of tickers.

        :param yfinance: Whether to use yfinance data. Default is True.
        :return: A tuple containing a list of filtered stocks, the dataframe of read data, 
        and the dataframe from B3.
        """
        
        if local_doc:
            home = Path.home()
            file = Path(home, 'Documents', 'GitHub', 'database', 'setorial_b3.xlsx')
        else:
            file = glob(self.tmp_path + '/' + 'Setorial*.xlsx')[0]
        
        df_b3 = pd.read_excel(file)
        df_b3 = (df_b3.ffill().dropna(axis=0))
        
        # Select only rows containing N1, N2, or NM
        select_units = df_b3.iloc[:,-1].str.contains('/(?<![\w\d])[NM]|(?<![\w\d])[N1]|(?<![\w\d])[N2]/', regex=True)
        df_b3 = df_b3[select_units]
        
        df_b3.dropna(thresh=df_b3.shape[1]*0.6, axis=0, inplace=True)
        cols = ['sector', 'subsector', 'name', 'ticker', 'market']
        df_b3.columns = cols
        
        # B3 Stocks
        filter_list = [i for i in list(df_b3.ticker) if i != 'LISTAGEM']
        
        # Enfoque database stocks
        enfoque_path = Path(Path.home(), 'Documents', 'GitHub', 'database', 'mercadoavista.csv')
        enfoque_read = pd.read_csv(enfoque_path, sep=';', decimal=',', encoding='Latin-1', header=None)
        cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 
                'Trades', 'Financial_Volume', 'Uncertain']
        enfoque_read.columns = cols
        enfoque_read = enfoque_read[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Financial_Volume']]
        
        etfs = pd.read_excel(self.assets_path, sheet_name='ETF')
        fiis = pd.read_excel(self.assets_path, sheet_name='FII Completo')
        fiis = fiis.applymap(self.add_11)
        fiis_manual = complete_b3_fiis()
        dis = pd.read_excel(self.assets_path, sheet_name='DI')
        
        # Manual filter
        list_out_number = ['34', '35', '12', '3B', '39', '11B', '5G']
        list_out_tickers = list(etfs.Codigos) + list(fiis.Codigos) + list(dis.Codigos) + fiis_manual + [
            'XMAL11', 'XPCA11', 'XPID11', 'XPIE11', 'VIGT11', 'VGIA11', 'VFDL11', 'IBOV11',
            'HASH11', 'BOVX11', 'ABCB10', 'ABCB10L', 'AHEB5', 'AHEB6', 'ALUG11', 'URET11',
            'USAL11', 'TORM13', 'THRA11', 'TEQI11', 'STOC31', 'SPXB11', 'SHOT11', 'RZAG11',
            'RURA11', 'REVE11', 'RBRI11', 'RBIV11', 'RBBV11', 'QIFF11', 'QETH11', 'QDFI11',
            'QBTC11', 'PPLA11', 'PPEI11', 'PICE11', 'LOGN3L', 'KIVO11', 'KDIF11', 'JURO11',
            'JGPX11', 'ITIT11', 'ITIP11', 'IRFM11', 'IMBB11', 'IGTI11', 'IFRA11', 'IBOB11',
            'IB5M11', 'HMOC11', 'HDOF11', 'GURU11', 'FSTU11', 'FSRF11', 'FSPE11', 'FPOR11',
            'FNOR11', 'FNAM11', 'FGAA11', 'FCCQ11', 'ETHE11', 'ENDD11', 'ELAS11', 'DEFI11',
            'LINX3', 'IBXL11', 'TASA17', 'TASA15', 'clsc4', 'YOUC3', 'YDRO11']
        
        filtered_assets = [asset for asset in list(enfoque_read['Ticker'])
                           if not any(x in asset for x in list_out_number)
                           and asset not in list_out_tickers]
        
        enfoque_read = enfoque_read[enfoque_read.Ticker.isin(filtered_assets)]
    
        enfoque_list = list(enfoque_read.Ticker.unique())
        
        # Final list
        check_list = [i for i in enfoque_list if i[0:4] in filter_list]
        
        # Undesired assets
        filtered_stocks = [i for i in check_list if not i.endswith(('10', '12', '15', '17', '3T', '3L', '34'))]
        
        return filtered_stocks, enfoque_read, df_b3


def main():
    tmp_path = tempfile.mkdtemp()
    t = Tickers_B3(tmp_path)
    t.open_file()
    assets, prices_enfoque, sectors_b3 = t.get_tickers()
    
    print("Filtered Stocks:")
    print(assets)
    print("\nEnfoque Read Dataframe:")
    print(prices_enfoque.head())
    print("\nB3 Sectors:")
    print(sectors_b3.head())

if __name__ == "__main__":
    main()
