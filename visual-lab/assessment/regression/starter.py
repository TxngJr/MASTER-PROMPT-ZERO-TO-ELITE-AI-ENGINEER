"""เขียนด้วย NumPy ก่อน อย่านำไฟล์ QA reference มาใช้แทนผลงานของตัวเอง.

X:(n,d), y:(n,), w:(d,), b:scalar; n>0 และข้อมูลต้อง finite.
ใช้ J=mean((Xw+b-y)**2)+lam*sum(w**2), ไม่ลงโทษ b.
reject shape ผิด/empty ด้วย ValueError; lam, eta, l1,l2 ต้องไม่ติดลบ.
ฟังก์ชันต้องไม่แก้ array ของ caller.
"""

def predict(X, w, b):
    """คืน predictions shape (n,)."""
    raise NotImplementedError

def loss(X, y, w, b, lam=0.0):
    """คืน J เป็น scalar ตามนิยาม mean loss ด้านบน."""
    raise NotImplementedError

def gradients(X, y, w, b, lam=0.0):
    """คืน (dw:(d,), db:scalar) จากสถานะเดิมเดียวกัน."""
    raise NotImplementedError

def update(X, y, w, b, eta, lam=0.0):
    """คืน (new_w,new_b) หลัง simultaneous gradient step หนึ่งครั้ง."""
    raise NotImplementedError

def fit_gd(X, y, eta=0.05, steps=2000):
    """เริ่ม w=0,b=0; คืน (w,b) หลังฝึก MSE; ไม่รวม regularization."""
    raise NotImplementedError

def fit_ols(X, y):
    """คืน (w,b) ของ least squares; รองรับ rank deficient โดยไม่ inverse ตรง."""
    raise NotImplementedError

def fit_ridge(X, y, lam):
    """คืน (w,b) ของ J=MSE+lam*L2; ไม่ลงโทษ intercept."""
    raise NotImplementedError

def polynomial_features(X, degree):
    """ทุก monomial degree 1..degree ตาม sklearn PolynomialFeatures(include_bias=False).
    degree ต้องเป็น integer >=1; ตัวอย่าง [a,b] degree2 -> [a,b,a*a,a*b,b*b].
    """
    raise NotImplementedError

def metrics(y, pred):
    """คืน dict mse,mae,rmse,r2. constant y: r2=1 เมื่อ exact,0 เมื่อไม่ exact."""
    raise NotImplementedError

def standardize_train(X_train, X_other):
    """คืน (train_scaled,other_scaled,mean,scale) ใช้ train statistics เท่านั้น.
    std ใช้ ddof=0 และ column std=0 ให้ scale=1; ห้าม fit บน X_other.
    """
    raise NotImplementedError

def fit_lasso(X, y, lam):
    """คืน (w,b) minimize MSE+lam*L1; coordinate descent/proximal ใช้ได้.
    ตรวจ KKT tolerance 2e-4; max iterations/tolerance กำหนดภายในโดยอธิบายในรายงาน.
    """
    raise NotImplementedError

def fit_elastic_net(X, y, l1, l2):
    """คืน (w,b) minimize MSE+l1*L1+l2*L2; bias ไม่รับ penalty."""
    raise NotImplementedError
