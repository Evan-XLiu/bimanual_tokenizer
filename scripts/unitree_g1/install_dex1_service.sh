#!/bin/sh
# Run once as root on the robot. Grant the fixed helper and exact Dex1 restart.
set -eu
test "$(id -u)" = 0
test -f /etc/systemd/system/dex1_gripper.service
command -v visudo >/dev/null
install -d -o root -g root -m 755 /usr/local/sbin
cat > /usr/local/sbin/g1-ensure-dex1 <<'HELPER'
#!/bin/sh
set -eu
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
test "$#" = 0
test "$(id -u)" = 0
unit=/etc/systemd/system/dex1_gripper.service
dir=/etc/systemd/system/dex1_gripper.service.d
test -f "$unit"
install -d -o root -g root -m 755 "$dir"
if [ ! -e "$dir/original-unit.backup" ]; then
    cp -p "$unit" "$dir/original-unit.backup"
fi
tmp=$(mktemp "$dir/.ordering.XXXXXX")
trap 'rm -f "$tmp"' EXIT
# Ordering dependencies cannot be cleared through a drop-in. Change only the
# known vendor line in the full unit, preserving ExecStart and all other fields.
sed 's/^After=multi-user.target$/After=basic.target systemd-udev-trigger.service/' "$unit" > "$tmp"
if ! cmp -s "$tmp" "$unit"; then
    install -o root -g root -m 644 "$tmp" "$unit"
fi
# Replace the previous ineffective reset with an additive-only drop-in.
printf '[Unit]\nAfter=basic.target systemd-udev-trigger.service\n' > "$tmp"
if ! cmp -s "$tmp" "$dir/90-g1-startup.conf"; then
    install -o root -g root -m 644 "$tmp" "$dir/90-g1-startup.conf"
fi
# Also reload after an interrupted previous installation.
systemctl daemon-reload
if systemctl show dex1_gripper.service -p After --value | tr ' ' '\n' | grep -qx multi-user.target; then
    echo 'Dex1 still depends on multi-user.target; inspect other overrides' >&2
    exit 1
fi
if ! systemctl is-active --quiet dex1_gripper.service; then
    timeout 25 systemctl start dex1_gripper.service
fi
systemctl is-active --quiet dex1_gripper.service
systemctl show dex1_gripper.service -p ActiveState -p SubState -p After
HELPER
chown root:root /usr/local/sbin/g1-ensure-dex1
chmod 755 /usr/local/sbin/g1-ensure-dex1
policy=$(mktemp)
trap 'rm -f "$policy"' EXIT
printf 'unitree ALL=(root) NOPASSWD: /usr/local/sbin/g1-ensure-dex1 ""\n' > "$policy"
printf 'unitree ALL=(root) NOPASSWD: /usr/bin/systemctl restart dex1_gripper.service\n' >> "$policy"
visudo -cf "$policy"
install -o root -g root -m 440 "$policy" /etc/sudoers.d/g1-dex1
/usr/local/sbin/g1-ensure-dex1
