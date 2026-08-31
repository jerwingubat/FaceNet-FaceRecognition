import os
import csv
from datetime import datetime, date

ATTENDANCE_FILE = "attendance.csv"
COLUMNS = ["Name", "Date", "TimeIn", "TimeOut", "Status", "Duration"]

# Attendance schedule threshold (24h). Set to None to disable late detection.
LATE_THRESHOLD = datetime.strptime("08:00", "%H:%M").time()


def _ensure_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)


def _load_records():
    _ensure_file()
    records = []
    if not os.path.exists(ATTENDANCE_FILE):
        return records
    with open(ATTENDANCE_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Name"):
                records.append({
                    "Name": row["Name"],
                    "Date": row.get("Date", ""),
                    "TimeIn": row.get("TimeIn", ""),
                    "TimeOut": row.get("TimeOut", ""),
                    "Status": row.get("Status", ""),
                    "Duration": row.get("Duration", ""),
                })
    return records


def _save_records(records):
    _ensure_file()
    with open(ATTENDANCE_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def _compute_duration(time_in, time_out):
    try:
        fmt = "%H:%M:%S"
        tin = datetime.strptime(time_in, fmt)
        tout = datetime.strptime(time_out, fmt)
        hours = (tout - tin).total_seconds() / 3600.0
        if hours < 0:
            hours += 24
        return f"{hours:.2f}"
    except ValueError:
        return ""


def check_in(name, now=None):
    if now is None:
        now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    records = _load_records()

    status = "Present"
    if LATE_THRESHOLD is not None:
        if now.time() > LATE_THRESHOLD:
            status = "Late"

    for rec in records:
        if rec["Name"] == name and rec["Date"] == day and not rec["TimeIn"]:
            rec["TimeIn"] = time_str
            rec["Status"] = status
            _save_records(records)
            return rec

    for rec in records:
        if rec["Name"] == name and rec["Date"] == day:
            # Already checked in for today
            return rec

    records.append({
        "Name": name,
        "Date": day,
        "TimeIn": time_str,
        "TimeOut": "",
        "Status": status,
        "Duration": "",
    })
    _save_records(records)
    return records[-1]


def check_out(name, now=None):
    if now is None:
        now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    records = _load_records()
    changed = False
    for rec in records:
        if rec["Name"] == name and rec["Date"] == day and rec["TimeIn"] and not rec["TimeOut"]:
            rec["TimeOut"] = time_str
            rec["Duration"] = _compute_duration(rec["TimeIn"], rec["TimeOut"])
            if rec["Status"] not in ("Late", "Present"):
                rec["Status"] = "Complete"
            changed = True
            break
    if changed:
        _save_records(records)
    return changed


def get_today_records(day=None):
    if day is None:
        day = date.today().strftime("%Y-%m-%d")
    return [r for r in _load_records() if r["Date"] == day]


def get_open_records(day=None):
    records = get_today_records(day)
    return [r for r in records if r["TimeIn"] and not r["TimeOut"]]


if __name__ == "__main__":
    _ensure_file()
    print(f"Attendance file: {os.path.abspath(ATTENDANCE_FILE)}")
    today = date.today().strftime("%Y-%m-%d")
    for r in get_today_records(today):
        print(r)
