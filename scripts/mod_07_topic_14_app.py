import requests
import streamlit as st

# Адреса сервісу FastAPI, який повертає прогноз
API_URL = 'http://127.0.0.1:8000/predict'

st.title('Прогноз продажів BigMart')

# Форма збирає ознаки товару й магазину; значення за замовчуванням — перший товар
# із набору даних. Запит надсилається лише після натискання кнопки
with st.form('item'):
    item = {
        'Item_Weight': st.number_input('Вага товару', value=9.3),
        'Item_Fat_Content': st.selectbox(
            'Вміст жиру', ['Low Fat', 'Regular']),
        'Item_Visibility': st.number_input(
            'Частка площі викладки', value=0.016, format='%.3f'),
        'Item_Type': st.selectbox(
            'Категорія товару',
            ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy',
             'Frozen Foods', 'Fruits and Vegetables', 'Hard Drinks',
             'Health and Hygiene', 'Household', 'Meat', 'Others',
             'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'],
            index=4),
        'Item_MRP': st.number_input('Ціна товару', value=249.81),
        'Outlet_Identifier': st.selectbox(
            'Магазин',
            ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019',
             'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'],
            index=9),
        'Outlet_Establishment_Year': st.number_input(
            'Скільки років працює магазин', value=14, step=1),
        'Outlet_Size': st.selectbox(
            'Розмір магазину', ['Small', 'Medium', 'High'], index=1),
        'Outlet_Location_Type': st.selectbox(
            'Тип місцевості', ['Tier 1', 'Tier 2', 'Tier 3']),
        'Outlet_Type': st.selectbox(
            'Тип магазину',
            ['Grocery Store', 'Supermarket Type1',
             'Supermarket Type2', 'Supermarket Type3'],
            index=1)}

    submitted = st.form_submit_button('Отримати прогноз')

# Застосунок не містить моделі: він надсилає ознаки до FastAPI у форматі JSON
# і показує число, яке повернув сервіс
if submitted:
    response = requests.post(API_URL, json=item, timeout=30)
    st.metric(label='Очікувані продажі',
              value=f"{response.json()['Item_Outlet_Sales']:.2f}")
