# JNI entry points and the classes the native code constructs by name.
-keep class com.rishikesh.edgechat.engine.NativeEngine { *; }
-keep class com.rishikesh.edgechat.engine.NativeModelInfo { *; }
-keep class com.rishikesh.edgechat.engine.NativeStats { *; }
-keep class com.rishikesh.edgechat.engine.EngineException { *; }
-keep interface com.rishikesh.edgechat.engine.NativeEngine$Listener { *; }
-keep interface com.rishikesh.edgechat.engine.NativeEngine$Progress { *; }
-keep interface com.rishikesh.edgechat.engine.NativeEngine$Logger { *; }
-keepclassmembers class * implements com.rishikesh.edgechat.engine.NativeEngine$Listener { *; }
-keepclassmembers class * implements com.rishikesh.edgechat.engine.NativeEngine$Progress { *; }
-keepclassmembers class * implements com.rishikesh.edgechat.engine.NativeEngine$Logger { *; }

# kotlinx.serialization
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt
-keepclassmembers class kotlinx.serialization.json.** { *** Companion; }
-keepclasseswithmembers class kotlinx.serialization.json.** { kotlinx.serialization.KSerializer serializer(...); }
-keep,includedescriptorclasses class com.rishikesh.edgechat.**$$serializer { *; }
-keepclassmembers class com.rishikesh.edgechat.** { *** Companion; }
-keepclasseswithmembers class com.rishikesh.edgechat.** { kotlinx.serialization.KSerializer serializer(...); }

# PDFBox-Android pulls in optional libraries we don't ship.
-dontwarn com.gemalto.jp2.**
-dontwarn org.bouncycastle.**
-dontwarn javax.**
