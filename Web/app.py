from flask import Flask, request, render_template, send_file

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search')
def search():
    query = request.args.get('query')
    # 模拟搜索结果
    results = [
        {"title": "Result 1", "url": "https://example.com/1"},
        {"title": "Result 2", "url": "https://example.com/2"},
        {"title": "Result 3", "url": "https://example.com/3"}
    ]
    return render_template('results.html', query=query, results=results)


@app.route('/Player1')
def players():
    return render_template('Player1.html')


@app.route('/Data')
def my_database():
    csv_path = "../database/generate_data/all_data_processed4_pro.csv"
    return send_file(csv_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
