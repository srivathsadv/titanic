import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from titanic_model.config.core import config
from titanic_model.processing.data_manager import load_raw_dataset
from sklearn.model_selection import train_test_split

def test_age_variable_transformer():
    
    data = load_raw_dataset(file_name=config.app_config_.training_data_file)

    X = data.drop(config.model_config_.target, axis=1)       # predictors
    y = data[config.model_config_.target]                    # target

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,  # predictors
        y,  # target
        test_size=config.model_config_.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config_.random_state,
    )
    
    print(X_test)
    print(X_test.loc[709])
    
if __name__ == "__main__":

    test_age_variable_transformer()
