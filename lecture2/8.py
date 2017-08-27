import tensorflow as tf

# InteractiveSession
# makes itself the default session and runs/evals without explicitly calling the session

sess = tf.InteractiveSession()

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# We can just use 'c.eval()' without passing 'sess'
print(c.eval())
sess.close() # closes the session

# tf.get_default_session() resutns the default session and it will be the innermost session on which a Session or Session.as_default() context has been entered

