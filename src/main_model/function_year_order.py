import matplotlib.pyplot as plt
import pandas as pd


def prepare_data_beta(data):
    X = data[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F11', 'F12']]
    y = data['L1']
    return X, y


def show_tendency(results):
    years = [start_year for start_year, accuracy in results]
    accuracies = [accuracy for start_year, accuracy in results]

    # 绘制趋势图
    plt.figure(figsize=(10, 6))
    plt.plot(years, accuracies, marker='o', linestyle='-', color='b')

    # 添加标题和标签
    plt.title('Accuracy Trend Over Different Year Ranges')
    plt.xlabel('Start Year')
    plt.ylabel('Accuracy')

    # 添加网格和显示图形
    plt.grid(True)
    plt.xticks(years)
    plt.ylim(0, 1)  # 假设准确率在0到1之间
    plt.show()


def predict_new_data_beta(voting_classifier, X_test, y_test):
    # Load new data from a CSV file

    if isinstance(X_test, pd.Series):
        X_test = X_test.to_frame()
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_frame(name='target')  # 将 y_test 转换为 DataFrame 并命名为 'target'
    new_data = pd.concat([X_test, y_test], axis=1)

    print(new_data)

    # Select features for prediction
    X_new = new_data[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F11', 'F12']]

    # Predict using the trained voting classifier
    new_predictions = voting_classifier.predict(X_new)

    # Add predictions to the new data
    new_data['L1'] = new_predictions

    # Print the predictions
    print(new_data[['Predicted Olympic Ranking']])

    # Save the predictions to a new CSV file
    new_data.to_csv('candidate_predictions_year_order.csv', index=False)
