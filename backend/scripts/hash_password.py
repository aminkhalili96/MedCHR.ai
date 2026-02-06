from getpass import getpass

from backend.app.security import hash_password


def main() -> None:
    password = getpass("Enter password to hash: ")
    confirm = getpass("Confirm password: ")
    if password != confirm:
        raise SystemExit("Passwords do not match.")
    print(hash_password(password))


if __name__ == "__main__":
    main()
