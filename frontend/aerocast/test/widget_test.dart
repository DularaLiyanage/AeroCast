import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:aerocast/main.dart';

void main() {
  testWidgets('App loads smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const AeroCastApp());
    
    // Wait for animations (like flutter_animate) to finish to avoid pending timer errors
    await tester.pumpAndSettle();

    // Verify that the splash screen or home screen is rendered.
    // The app should at least have a MaterialApp and a Scaffold.
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}

