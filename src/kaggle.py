
import kaggle

class Kaggle:
    def __init__(self) -> None:    
        self.authenticated = False
        

    def authenticate(self):
        kaggle.api.authenticate()
        self.authenticated = True
        print("Authenticated")
        
        return self
        
    
    def download_data(self):
        """ Downloads data from kaggle API """
        if self.authenticated:
            kaggle.api.dataset_download_files(
                "nopdev/real-and-fake-news-dataset/", path="real_fake", unzip=True
            )
            print("Finished downloading...")