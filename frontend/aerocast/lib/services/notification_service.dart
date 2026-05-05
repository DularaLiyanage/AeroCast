import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter/material.dart';

class NotificationService {
  static final FlutterLocalNotificationsPlugin _notificationsPlugin =
      FlutterLocalNotificationsPlugin();

  static Future<void> initialize() async {
    const AndroidInitializationSettings initializationSettingsAndroid =
        AndroidInitializationSettings('@mipmap/ic_launcher');

    // Currently only focused on Android as requested tests are mapped for Emulator
    const InitializationSettings initializationSettings =
        InitializationSettings(
      android: initializationSettingsAndroid,
    );

    // Request permissions for Android 13+
    await _notificationsPlugin
        .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>()
        ?.requestNotificationsPermission();

    await _notificationsPlugin.initialize(
      settings: initializationSettings,
      onDidReceiveNotificationResponse: (NotificationResponse response) async {
        // Handle notification tapped
      },
    );
  }

  static Future<void> showAqiAlert({
    required String title,
    required String message,
    int id = 0,
  }) async {
    const AndroidNotificationDetails androidPlatformChannelSpecifics =
        AndroidNotificationDetails(
      'aqi_alerts',
      'AQI Alerts',
      channelDescription: 'Notifications for high AQI risk levels',
      importance: Importance.max,
      priority: Priority.high,
      icon: '@mipmap/ic_launcher',
      color: Colors.redAccent, // Adjust color as needed based on risk
      enableVibration: true,
      playSound: true,
    );

    const NotificationDetails platformChannelSpecifics =
        NotificationDetails(android: androidPlatformChannelSpecifics);

    await _notificationsPlugin.show(
      id: id,
      title: title,
      body: message,
      notificationDetails: platformChannelSpecifics,
    );
  }

  static Future<void> showTestNotification() async {
    await showAqiAlert(
      title: 'TEST: High AQI Risk',
      message: 'Air Quality is unhealthy. Please take precautions.',
      id: 999,
    );
  }
}
