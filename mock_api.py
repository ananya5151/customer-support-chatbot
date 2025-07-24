from flask import Flask, jsonify

app = Flask(__name__)

# This is our fake order database
mock_orders = {
    "ORD12345": {"status": "Shipped", "estimated_delivery": "2 days"},
    "ORD67890": {"status": "Processing", "estimated_delivery": "5 days"},
    "ORD54321": {"status": "Delivered", "estimated_delivery": "N/A"}
}

@app.route('/order_status/<order_id>', methods=['GET'])
def get_order_status(order_id):
    order = mock_orders.get(order_id)
    if order:
        return jsonify(order)
    else:
        return jsonify({"error": "Order not found"}), 404

if __name__ == '__main__':
    # Runs the API on http://127.0.0.1:5000
    app.run(port=5000, debug=True)