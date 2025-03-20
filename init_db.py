from app import app, db, User

# Create tables and initial admin user
with app.app_context():
    db.create_all()

    # Check if admin user exists
    if not User.query.filter_by(username="admin").first():
        admin = User(
            username="admin",
            email="kedamm@microsoft.com",
            is_admin=True,
            is_approved=True,
        )
        admin.set_password("admin123")
        db.session.add(admin)

        # Create a regular user for testing
        user = User(username="user", email="user@example.com", is_approved=True)
        user.set_password("user123")
        db.session.add(user)

        db.session.commit()
        print("Database initialized with admin and test user")
    else:
        print("Admin user already exists")
