import 'package:flutter/material.dart';
import '../utils/constants.dart';

class CustomCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;
  final Color? color;
  
  const CustomCard({
    super.key, 
    required this.child, 
    this.padding = const EdgeInsets.all(16.0),
    this.color = Colors.white,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(AppStyles.cardRadius),
        boxShadow: [AppStyles.softShadow],
      ),
      child: Padding(
        padding: padding,
        child: child,
      ),
    );
  }
}
