import pandas as pd

# 创建示例数据
data = {
    'android_res_id': ['splash_slogan', 'login_by_facebook', 'login_by_apple', 'login_by_google', 'login_user_protocol'],
    '中文': ['当我们热爱世界时，我们就生活在其中。', '用Facebook登录', '用Apple登录', '用Google登录', '继续表示同意我们的用户协议和隐私政策'],
    '繁體中文': ['When we love to world,we live in it.', '用Facebook登錄', '用Apple登錄', '用Google登錄', '繼續表示同意我們的用戶協議和隱私政策'],
    'English': ['When we love to world,we live in it.', 'Log in with Facebook', 'Log in with Apple', 'Log in with Google', 'By continue, you agree to the User Agreement and Privacy Police.'],
    'EN-审校稿': ['We live in the world when we love it.', 'Log in with Facebook', 'Log in with Apple', 'Log in with Google', 'By continuing, you agree to the User Agreement and Privacy Policy.'],
    '日语-审校稿': ['私たちが世界を愛するとき、私たちはその中で生きているのです。', '즐거운 생활의 시작, ShortTV', 'Appleでログインする', 'Googleでログインする', '続けるには、利用規約およびプライバシーポリシーに同意する必要があります。'],
    '韩语-审校稿': ['즐거운 생활의 시작, ShortTV', '페이스북으로 로그인하기', 'Apple로 로그인하기', 'Google로 로그인하기', '계속 진행하면 이용약관과 개인정보처리방침에 동의하는 것을 의미합니다.'],
    '泰语-审校稿': ['เมื่อเรารักโลก เราก็อยู่ในนั้น', 'เข้าสู่ระบบด้วย Facebook', 'Apple로 로그인하기', 'Google로 로그인하기', 'เมื่อดำเนินการต่อไป คุณตกลงยินยอมตามข้อกำหนดและนโยบายความเป็นส่วนตัวของผู้ใช้']
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 将DataFrame保存为CSV文件
csv_file_path = 'data_2.csv'
df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 编码避免 Excel 中文乱码

print(f'CSV文件已创建：{csv_file_path}')